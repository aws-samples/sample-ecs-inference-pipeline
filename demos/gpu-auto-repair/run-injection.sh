#!/usr/bin/env bash
#
# run-injection.sh — inject a synthetic NVIDIA XID error into a GPU on an ECS
# Managed Instances container instance to demonstrate GPU auto repair.
#
# It registers the xid-inject task definition and runs it on a GPU container
# instance via the GPU capacity provider. The task bind-mounts the host's DCGM
# socket directory (/run/nvidia-dcgm) and uses DCGM's error-injection framework
# (dcgmi test --inject -f 230 --host unix:///hostdcgm/nv-hostengine) to write a
# synthetic XID into the SAME nv-hostengine the ECS agent's GPU health monitor
# reads. No real hardware fault is required.
#
# One critical XID on one GPU is a complete repair test: ACCELERATED_COMPUTE is
# an instance-level health check, so the first critical XID marks the whole
# instance IMPAIRED and (with auto repair enabled) triggers replacement.
#
# NOTE on instance targeting: Amazon ECS Managed Instances does NOT allow
# pinning a task to a specific ec2InstanceId (placement-constraint expressions
# are limited to ecs.subnet-id/vpc-id/availability-zone/cpu-architecture/
# instance-type). The injector therefore lands on whichever GPU instance the
# capacity provider places it on. The script prints which instance that is.
#
# Usage:
#   STACK_NAME=ecs-gpu-inference AWS_REGION=us-west-2 ./run-injection.sh [XID_CODE] [GPU_ID]
#
# Args (all optional):
#   XID_CODE   NVIDIA XID to inject. Default 79 (GPU fell off the bus -> Replace).
#   GPU_ID     DCGM GPU index on the instance. Default 0.
#
set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION (e.g. us-west-2)}"
: "${STACK_NAME:?Set STACK_NAME (e.g. ecs-gpu-inference)}"

XID_CODE="${1:-79}"
GPU_ID="${2:-0}"

CLUSTER="${STACK_NAME}-cluster"
CAPACITY_PROVIDER="${CAPACITY_PROVIDER:-gpu-inference-${STACK_NAME}-capacity}"
TASKDEF_FILE="$(dirname "$0")/xid-inject-taskdef.json"
LOG_GROUP="/ecs/xid-inject"

echo "==> Cluster: ${CLUSTER} | capacity provider: ${CAPACITY_PROVIDER} (region ${AWS_REGION})"

# --- Resolve the IAM roles the stack already created (reused for the injector) ---
TASK_ROLE_NAME="${STACK_NAME}-task-role"
TASK_ROLE_ARN="$(aws iam get-role --role-name "${TASK_ROLE_NAME}" \
  --query 'Role.Arn' --output text 2>/dev/null || echo "")"
EXEC_ROLE_ARN="$(aws iam get-role --role-name "${STACK_NAME}-task-execution-role" \
  --query 'Role.Arn' --output text 2>/dev/null || echo "")"

if [[ -z "${TASK_ROLE_ARN}" || "${TASK_ROLE_ARN}" == "None" ]]; then
  echo "ERROR: could not resolve ${TASK_ROLE_NAME}. Set TASK_ROLE_ARN/EXEC_ROLE_ARN env vars manually." >&2
  exit 1
fi
: "${EXEC_ROLE_ARN:=${TASK_ROLE_ARN}}"

# --- Grant ECS Exec (SSM) permissions to the task role so on-demand injection works ---
# The stack's task role is scoped to S3/SQS; ECS Exec needs the ssmmessages channel.
aws iam put-role-policy \
  --role-name "${TASK_ROLE_NAME}" \
  --policy-name "EcsExecForXidInject" \
  --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["ssmmessages:CreateControlChannel","ssmmessages:CreateDataChannel","ssmmessages:OpenControlChannel","ssmmessages:OpenDataChannel"],"Resource":"*"}]}' \
  >/dev/null 2>&1 || echo "WARN: could not attach ECS Exec policy to ${TASK_ROLE_NAME}; on-demand exec may fail."

# --- Ensure the log group exists ---
aws logs create-log-group --log-group-name "${LOG_GROUP}" --region "${AWS_REGION}" 2>/dev/null || true

# --- Register the task definition with placeholders substituted ---
RENDERED="$(mktemp)"
sed -e "s#TASK_ROLE_ARN#${TASK_ROLE_ARN}#g" \
    -e "s#EXECUTION_ROLE_ARN#${EXEC_ROLE_ARN}#g" \
    -e "s#AWS_REGION#${AWS_REGION}#g" \
    "${TASKDEF_FILE}" > "${RENDERED}"

echo "==> Registering task definition 'xid-inject'..."
TASKDEF_ARN="$(aws ecs register-task-definition --cli-input-json "file://${RENDERED}" \
  --region "${AWS_REGION}" --query 'taskDefinition.taskDefinitionArn' --output text)"
rm -f "${RENDERED}"
echo "    ${TASKDEF_ARN}"

# --- Run the injector via the GPU capacity provider (Managed Instances) ---
echo "==> Running injector task (XID_CODE=${XID_CODE}, GPU_ID=${GPU_ID})..."
TASK_ARN="$(aws ecs run-task \
  --cluster "${CLUSTER}" \
  --task-definition xid-inject \
  --capacity-provider-strategy "capacityProvider=${CAPACITY_PROVIDER},weight=1,base=0" \
  --enable-execute-command \
  --count 1 \
  --overrides "{\"containerOverrides\":[{\"name\":\"xid-inject\",\"environment\":[{\"name\":\"XID_CODE\",\"value\":\"${XID_CODE}\"},{\"name\":\"GPU_ID\",\"value\":\"${GPU_ID}\"}]}]}" \
  --region "${AWS_REGION}" \
  --query 'tasks[0].taskArn' --output text)"

echo "==> Injector task: ${TASK_ARN}"

# --- Report which instance it landed on ---
echo "==> Waiting for placement..."
for _ in $(seq 1 20); do
  CI_ARN="$(aws ecs describe-tasks --cluster "${CLUSTER}" --tasks "${TASK_ARN}" \
    --region "${AWS_REGION}" --query 'tasks[0].containerInstanceArn' --output text 2>/dev/null || echo "")"
  [[ -n "${CI_ARN}" && "${CI_ARN}" != "None" ]] && break
  sleep 5
done
if [[ -n "${CI_ARN}" && "${CI_ARN}" != "None" ]]; then
  TARGET_INSTANCE="$(aws ecs describe-container-instances --cluster "${CLUSTER}" \
    --container-instances "${CI_ARN}" --region "${AWS_REGION}" \
    --query 'containerInstances[0].ec2InstanceId' --output text)"
  echo "==> Injector landed on: ${TARGET_INSTANCE}"
fi

echo ""
echo "Follow the injection log (expect 'Successfully injected field info.' / INJECT_OK):"
echo "  aws logs tail ${LOG_GROUP} --follow --region ${AWS_REGION}"
echo ""
echo "Then watch the repair cycle (IMPAIRED -> DRAINING -> replaced, ~2 min to detect, ~8 min total):"
echo "  STACK_NAME=${STACK_NAME} AWS_REGION=${AWS_REGION} ./observe-repair.sh"
echo ""
echo "On-demand injection (task idles for 1h) via ECS Exec:"
echo "  aws ecs execute-command --cluster ${CLUSTER} --task ${TASK_ARN} \\"
echo "    --container xid-inject --interactive \\"
echo "    --command \"dcgmi test --inject --gpuid 0 -f 230 -v 74 --host unix:///hostdcgm/nv-hostengine\" --region ${AWS_REGION}"
