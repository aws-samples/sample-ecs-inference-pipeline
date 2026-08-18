#!/usr/bin/env bash
#
# setup-health-events.sh — create an EventBridge rule + CloudWatch Logs target
# that records every "ECS Container Instance Health Change" event where the
# instance becomes IMPAIRED. This gives you a durable, timestamped record of GPU
# XID faults and the repair actions that follow, alongside the live polling in
# observe-repair.sh.
#
# Usage:
#   AWS_REGION=us-west-2 ./setup-health-events.sh
#
set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION (e.g. us-west-2)}"

RULE_NAME="ecs-gpu-health-impaired"
LOG_GROUP="/ecs/container-instance-health"
PATTERN='{"source":["aws.ecs"],"detail-type":["ECS Container Instance Health Change"],"detail":{"overallStatus":["IMPAIRED"]}}'

echo "==> Creating log group ${LOG_GROUP}"
aws logs create-log-group --log-group-name "${LOG_GROUP}" --region "${AWS_REGION}" 2>/dev/null || true

echo "==> Creating EventBridge rule ${RULE_NAME}"
aws events put-rule \
  --name "${RULE_NAME}" \
  --event-pattern "${PATTERN}" \
  --region "${AWS_REGION}" >/dev/null

LOG_ARN="$(aws logs describe-log-groups --log-group-name-prefix "${LOG_GROUP}" \
  --region "${AWS_REGION}" --query "logGroups[?logGroupName=='${LOG_GROUP}'].arn | [0]" --output text)"

echo "==> Wiring the log group as the rule target"
aws events put-targets \
  --rule "${RULE_NAME}" \
  --targets "Id=cwlogs,Arn=${LOG_ARN%:\*}" \
  --region "${AWS_REGION}" >/dev/null

# EventBridge needs a resource policy on the log group to deliver events.
aws logs put-resource-policy \
  --policy-name "${RULE_NAME}-eb" \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":[\"events.amazonaws.com\",\"delivery.logs.amazonaws.com\"]},\"Action\":[\"logs:CreateLogStream\",\"logs:PutLogEvents\"],\"Resource\":\"${LOG_ARN%:\*}:*\"}]}" \
  --region "${AWS_REGION}" >/dev/null

echo "Done. Tail GPU health-change events with:"
echo "  aws logs tail ${LOG_GROUP} --follow --region ${AWS_REGION}"
