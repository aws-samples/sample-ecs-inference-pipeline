#!/usr/bin/env bash
#
# observe-repair.sh — watch the ECS Managed Instances GPU auto-repair cycle.
#
# Polls every ACTIVE/DRAINING container instance in the cluster and prints its
# ACCELERATED_COMPUTE health check (status + XID reason), overall status, and
# running task count. After a critical XID is injected you should see the
# instance flip to IMPAIRED (statusReason XID_<n>), then move to DRAINING while
# a replacement is provisioned (start-before-stop), then disappear.
#
# Usage:
#   STACK_NAME=ecs-gpu-inference AWS_REGION=us-west-2 ./observe-repair.sh [INTERVAL_SECONDS]
#
set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION (e.g. us-west-2)}"
: "${STACK_NAME:?Set STACK_NAME (e.g. ecs-gpu-inference)}"

CLUSTER="${STACK_NAME}-cluster"
INTERVAL="${1:-15}"

echo "Watching GPU health + repair on ${CLUSTER} every ${INTERVAL}s (Ctrl-C to stop)"
echo "Expected sequence: OK -> IMPAIRED (XID_<n>) -> DRAINING -> terminated; replacement launches."
echo ""

while true; do
  printf '===== %s =====\n' "$(date -u +%H:%M:%SZ)"

  CI_ARNS="$(aws ecs list-container-instances --cluster "${CLUSTER}" \
    --region "${AWS_REGION}" --query 'containerInstanceArns' --output text 2>/dev/null || echo "")"

  if [[ -z "${CI_ARNS}" || "${CI_ARNS}" == "None" ]]; then
    echo "  (no container instances registered)"
  else
    # shellcheck disable=SC2086
    aws ecs describe-container-instances \
      --cluster "${CLUSTER}" \
      --container-instances ${CI_ARNS} \
      --include CONTAINER_INSTANCE_HEALTH \
      --region "${AWS_REGION}" \
      --query 'containerInstances[].{ec2:ec2InstanceId,state:status,overall:healthStatus.overallStatus,accel:healthStatus.details[?type==`ACCELERATED_COMPUTE`]|[0].status,reason:healthStatus.details[?type==`ACCELERATED_COMPUTE`]|[0].statusReason,running:runningTasksCount}' \
      --output table
  fi

  echo ""
  sleep "${INTERVAL}"
done
