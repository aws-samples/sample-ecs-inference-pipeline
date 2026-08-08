# ECS GPU Inference Pipeline

Production-ready, dual-tier GPU inference pipeline on Amazon ECS Managed Instances with intelligent request routing. An API Gateway endpoint accepts inference requests, a Router Lambda classifies complexity and routes to the appropriate tier: Mistral-7B on g6e.xlarge for simple requests, Llama-2-70B on g6e.48xlarge for complex ones. Both tiers scale independently from zero, recover automatically from GPU hardware failures, and write results to S3 deployed with a single CloudFormation template.

## Architecture

![Dual-Tier GPU Inference Pipeline](generated-diagrams/ecs-gpu-inference-dual-tier.svg)

### Request Flow

1. Producer sends a JSON request to `POST /infer` on the API Gateway endpoint
2. Router Lambda classifies by complexity (prompt length, maxTokens, or explicit `"route"` override) and enqueues to the appropriate SQS queue
3. ECS tasks on GPU Managed Instances long-poll their respective queue via VPC endpoint
4. SQS worker validates messages, malformed ones go to a shared DLQ
5. Valid requests are forwarded to the local vLLM server for inference
6. Results are written to S3 as `results/{requestId}.json` with exponential-backoff retry
7. Processed messages are deleted from the queue; failures return through visibility timeout
8. Autoscaling adjusts each tier independently based on queue depth + GPU memory utilization

## Features

- **Intelligent routing**  Router Lambda classifies requests to the right-sized model tier (heuristic + optional Bedrock classification for ambiguous cases)
- **Scale-to-zero**  no GPU instances running when queues are empty; ECS MI terminates idle instances automatically
- **Independent scaling**  each tier has its own capacity provider, scaling metric, and alarms
- **GPU auto-repair**  ECS monitors NVIDIA GPU health via DCGM and auto-replaces impaired instances (XID errors 48, 74, 79, 95, 140)
- **Deployment circuit breaker**  automatically rolls back failed deployments on both services
- **Composite scaling metric**  `max(queueDepth/threshold, gpuUtilization/80)` per tier
- **Idempotent processing**  checks for existing results in S3 before running inference
- **Least-privilege IAM** separate task roles per tier, scoped to specific buckets, queues, and namespaces
- **Enhanced Container Insights**  GPU utilization, memory, temperature, power draw, and XID error telemetry
- **Pre-built CloudWatch dashboard**  4 sections: Routing, Small Tier, Large Tier, GPU Hardware
- **Automatic security patching**  14-day instance refresh using start-before-stop pattern

## Project Structure

```
├── infrastructure/
│   ├── template.yaml              # CloudFormation template (70 resources)
│   └── scaling_metric_lambda.py   # Composite scaling metric Lambda (queue depth + GPU utilization)
├── container/
│   ├── Dockerfile                 # vLLM v0.26.0 base + SQS worker
│   ├── entrypoint.sh              # Model download, vLLM startup, worker launch
│   ├── sqs_worker.py              # Queue polling, validation, inference forwarding
│   └── buildspec.yml              # CodeBuild spec for remote image builds
├── tests/
│   ├── conftest.py                # Shared pytest fixtures
│   ├── test_router_lambda.py      # Router Lambda routing logic
│   ├── test_sqs_worker_processing.py   # SQS worker end-to-end processing
│   ├── test_sqs_worker_validation.py   # Message schema validation
│   ├── test_sqs_long_polling.py        # Long-poll behavior
│   ├── test_valid_request_processing.py
│   ├── test_malformed_request_dlq_routing.py
│   ├── test_composite_scaling_metric.py
│   ├── test_model_s3_path_passthrough.py
│   ├── test_iam_least_privilege.py
│   ├── test_cross_references.py        # CloudFormation cross-resource reference checks
│   ├── test_integration.py             # End-to-end integration tests
│   ├── test_private_subnet_enforcement.py
│   ├── test_cost_allocation_tags.py
│   └── requirements.txt
├── generated-diagrams/
│   └── ecs-gpu-inference-dual-tier.svg # Architecture diagram
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── LICENSE
```

## Prerequisites

- AWS account with permissions for ECS, EC2, SQS, S3, Lambda, API Gateway, CloudWatch, IAM, CloudFormation
- AWS CLI v2 (v2.32.0+)
- Docker 20.10+ with buildx support (or use CodeBuild for remote builds — recommended for the ~8 GB image)
- Amazon EC2 GPU quota for g6e.xlarge and g6e.48xlarge in your target region
- Git, jq, Bash shell
- A quantized model on S3, or use HuggingFace direct download (default — no upload needed)

## Deployment Walkthrough

### Step 1: Clone Repository and Set Variables

```bash
git clone git@github.com:aws-samples/sample-ecs-inference-pipeline.git
cd sample-ecs-inference-pipeline

export AWS_REGION="us-west-2"
export STACK_NAME="ecs-gpu-inference"
export MODEL_NAME="TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
export LARGE_MODEL_NAME="TheBloke/Llama-2-70B-Chat-AWQ"

# Leave MODEL_S3_PATH empty to download from HuggingFace at runtime.
# Set to an S3 URI (e.g., s3://my-bucket/models/mistral-7b/) to load a pre-cached model.
export MODEL_S3_PATH=""
export LARGE_MODEL_S3_PATH=""
```

### Step 2: Deploy Infrastructure with CloudFormation

The template exceeds the 51 KB inline limit, so upload it to S3 first:

```bash
aws s3 mb s3://${STACK_NAME}-cfn-${AWS_REGION} --region $AWS_REGION
aws s3 cp infrastructure/template.yaml s3://${STACK_NAME}-cfn-${AWS_REGION}/template.yaml

aws cloudformation create-stack \
    --stack-name $STACK_NAME \
    --template-url https://${STACK_NAME}-cfn-${AWS_REGION}.s3.${AWS_REGION}.amazonaws.com/template.yaml \
    --parameters \
      ParameterKey=ModelS3Path,ParameterValue="s3://none/none/" \
      ParameterKey=ModelName,ParameterValue=$MODEL_NAME \
      ParameterKey=LargeModelS3Path,ParameterValue="s3://none/none/" \
      ParameterKey=LargeModelName,ParameterValue=$LARGE_MODEL_NAME \
    --capabilities CAPABILITY_NAMED_IAM \
    --region $AWS_REGION

aws cloudformation wait stack-create-complete --stack-name $STACK_NAME --region $AWS_REGION
```

> **Note:** Use `` as the S3 path placeholder when downloading models from HuggingFace. The entrypoint script skips S3 download for this value. Empty strings cause `Fn::Select` errors in the template.

Verify stack creation:

```bash
aws cloudformation describe-stacks \
  --stack-name $STACK_NAME \
  --query "Stacks[0].StackStatus" \
  --output text
# Expected: CREATE_COMPLETE
```

The template provisions 70 resources: VPC with private subnets and VPC endpoints, API Gateway, Router Lambda, SQS queues, ECS cluster with two managed-instance capacity providers (GPU auto-repair enabled), two services with independent scaling, least-privilege IAM roles, composite-metric autoscaling Lambdas, and a CloudWatch dashboard. 

### Step 3: Retrieve Stack Outputs

Export CloudFormation outputs as environment variables for use in subsequent steps:

```bash
export INFERENCE_API_URL=$(aws cloudformation describe-stacks --stack-name $STACK_NAME \
  --query 'Stacks[0].Outputs[?OutputKey==`InferenceApiUrl`].OutputValue' --output text)
export ECR_URI=$(aws cloudformation describe-stacks --stack-name $STACK_NAME \
  --query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryUri`].OutputValue' --output text)
export RESULTS_BUCKET=$(aws cloudformation describe-stacks --stack-name $STACK_NAME \ 
  --query 'Stacks[0].Outputs[?OutputKey==`ResultsBucketName`].OutputValue' --output text) 
export DASHBOARD_URL=$(aws cloudformation describe-stacks --stack-name $STACK_NAME \
  --query 'Stacks[0].Outputs[?OutputKey==`DashboardUrl`].OutputValue' --output text)
```

### Step 4: Build and Push the Inference Container

The container is based on `vllm/vllm-openai:v0.26.0` (pinned). All dependencies are pinned to exact versions. vLLM parameters are configurable through environment variables at runtime: MODEL_NAME, QUANTIZATION, MAX_SEQ_LEN, GPU_MEM_UTIL, TP_SIZE. The entrypoint handles model download from S3 (or HuggingFace if `MODEL_S3_PATH` is `s3://none/none/`), starts the vLLM OpenAI-compatible server, then launches the SQS worker.

**Container best practices callout:**

- **Pin the base image to an exact version tag** The inference landscape moves fast and breaking changes are common
- **Pin all dependency versions** no >= ranges
- **Build for a specific GPU architecture** Images are not portable across GPU types
- **Use --platform linux/amd64 explicitly** 
- **Pack light — only include strictly necessary dependencies** 
- **Tag with a versioned label (e.g., v2.0.0), not just :latest** 


**Option A CodeBuild (recommended for the ~8 GB image):**

```bash
cd container && zip -r /tmp/container-source.zip . && cd ..
aws s3 cp /tmp/container-source.zip s3://${STACK_NAME}-cfn-${AWS_REGION}/container-source.zip
aws codebuild start-build --project-name ${STACK_NAME}-build --region $AWS_REGION
```

**Option B Local build:**

```bash
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI
docker build --platform linux/amd64 -t $ECR_URI:v2.0.0 -t $ECR_URI:latest container/
docker push $ECR_URI:v2.0.0 && docker push $ECR_URI:latest
```

> **Timing:** The local build pulls ~8 GB for the vLLM base image and takes approximately 12 minutes to build, plus 20+ minutes to push all layers to ECR.

### Step 5: Deploy and Verify Both ECS Services

Force a new deployment on both services to pick up the pushed image:

```bash
SMALL_SVC=$(aws cloudformation describe-stacks --stack-name $STACK_NAME \
  --query 'Stacks[0].Outputs[?OutputKey==`SmallModelServiceName`].OutputValue' --output text)
LARGE_SVC=$(aws cloudformation describe-stacks --stack-name $STACK_NAME \
  --query 'Stacks[0].Outputs[?OutputKey==`LargeModelServiceName`].OutputValue' --output text)
CLUSTER=$(aws cloudformation describe-stacks --stack-name $STACK_NAME \
  --query 'Stacks[0].Outputs[?OutputKey==`ClusterArn`].OutputValue' --output text)

aws ecs update-service --cluster $CLUSTER --service $SMALL_SVC --force-new-deployment --region $AWS_REGION
aws ecs update-service --cluster $CLUSTER --service $LARGE_SVC --force-new-deployment --region $AWS_REGION
```

With empty queues, both services stabilize at 0 running tasks. This is expected: there are no messages to process, so the scaling metric stays at zero and no instances are provisioned.

The deployment circuit breaker is enabled on both services. If a deployment fails (container crashes during model load, health check timeout, OOM), the circuit breaker detects repeated failures and automatically rolls back to the previous working task definition. As of July 2026, you can configure the circuit breaker threshold using a fixed task failure count or percentage, and choose between consecutive failure counting (resets on success) or cumulative counting (failures accumulate). For GPU workloads with inherently long cold starts, a lower threshold with consecutive counting is recommended. GPU startup failures tend to be consistent rather than transient.

### Step 6: Test the Pipeline End-to-End

**Send a small-tier request:**

```bash
curl -X POST $INFERENCE_API_URL \
  -H "Content-Type: application/json" \
  -d '{
    "requestId": "550e8400-e29b-41d4-a716-446655440000",
    "prompt": "What is Amazon ECS? Answer briefly.",
    "maxTokens": 128,
    "temperature": 0.7
  }'
# → {"requestId": "550e8400-...", "tier": "small"}
```

The router responds synchronously with the assigned tier. Inference runs asynchronously and results arrive in S3.

Check results in S3:
```bash
aws s3 ls s3://$RESULTS_BUCKET/results/
aws s3 cp s3://$RESULTS_BUCKET/results/550e8400-e29b-41d4-a716-446655440000.json -{
  "requestId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "result": {
    "text": "Amazon ECS is a fully managed container orchestration service...",
    "usage": {"promptTokens": 17, "completionTokens": 102, "totalTokens": 119},
    "finishReason": "stop"
  },
  "processingTimeMs": 4956,
  "timestamp": "2026-05-03T04:35:47.906601+00:00"
}
```
Use - to output to stdout, or specify a local file path (e.g. ./result.json)


**Test large-tier explicit routing:**

```bash
curl -X POST $INFERENCE_API_URL \
  -H "Content-Type: application/json" \
  -d '{
    "requestId": "550e8400-e29b-41d4-a716-446655440001",
    "prompt": "Analyze the following contract and identify all liability clauses...",
    "maxTokens": 2048,
    "route": "large"
  }'
# → {"requestId": "550e8400-...", "tier": "large"}
```

**Trigger scale-out** by sending 6+ messages to exceed the `QueueDepthScaleThreshold` (default: 5):

```bash
for i in $(seq 2 7); do
  curl -s -X POST $INFERENCE_API_URL \
    -H "Content-Type: application/json" \
    -d "{
      \"requestId\": \"550e8400-e29b-41d4-a716-44665544000${i}\",
      \"prompt\": \"Explain GPU inference optimization technique number ${i}.\",
      \"maxTokens\": 128,
      \"temperature\": 0.7
    }"
done
```

The full cold-start sequence takes ~7 minutes:

1. Composite metric Lambda detects queue depth > threshold → metric exceeds 1.0
2. Scale-out alarm fires → step scaling adds 1 task
3. ECS Managed Instances provisions a g6e.xlarge (~5 min)
4. Container image pulled (~8 GB), vLLM downloads Mistral-7B-AWQ from HuggingFace
5. vLLM engine starts, health check passes, SQS worker begins polling
6. Queue drains: 6 messages processed in ~30 seconds once warm

**Observe scale-to-zero:** once the queue drains, the scale-in alarm evaluates for 5 consecutive periods below threshold (5 × 60s = 5 min for small tier), then scales the service back to 0 tasks. ECS MI terminates the idle instance.

**Open the dashboard** at `$DASHBOARD_URL` to view per-tier GPU utilization, queue depths, task counts, and routing metrics in real time.

### Step 7: Observe GPU Auto-Repair

GPU auto-repair runs automatically — no manual intervention required in production. To monitor instance health status:

```bash
# List container instances and their health status
aws ecs describe-container-instances \
  --cluster $CLUSTER \
  --container-instances $(aws ecs list-container-instances --cluster $CLUSTER \
    --query 'containerInstanceArns[]' --output text --region $AWS_REGION) \
  --query 'containerInstances[*].{Instance:ec2InstanceId,Status:status,Health:healthStatus.overallStatus,AgentConnected:agentConnected,RunningTasks:runningTasksCount}' \
  --output table --region $AWS_REGION
```

To receive notifications when an instance is drained for repair, subscribe to ECS container instance state change events through EventBridge:

```bash
aws events put-rule \
  --name ecs-gpu-instance-health \
  --event-pattern '{"source":["aws.ecs"],"detail-type":["ECS Container Instance State Change"],"detail":{"status":["DRAINING"]}}' \
  --region $AWS_REGION
```

When a critical XID error is detected (double-bit ECC, NVLink error, GPU fallen off bus), ECS sets the instance to DRAINING and provisions a replacement. Messages being processed return to the queue through visibility timeout and are picked up by the replacement instance.

## Routing Logic

| Signal | Routes to | Rationale |
|---|---|---|
| `"route": "large"` in body | Large tier | Explicit override, no further classification |
| `"route": "small"` in body | Small tier | Explicit override, no further classification |
| Prompt > 2,000 characters | Large tier | Long context implies complex reasoning (heuristic) |
| `maxTokens` > 1,024 | Large tier | Long-form generation benefits from 70B parameters |
| Prompt 500–2,000 chars, maxTokens ≤ 1,024 | Bedrock classification | Ambiguous zone: Nova Micro or Haiku determines intent |
| Default (< 500 chars) | Small tier | Most requests are simple enough for 7B |

## CloudFormation Parameters

| Parameter | Default | Description |
|---|---|---|
| `ModelS3Path` | `none` | S3 URI for small model (use `s3://none/none/` for HuggingFace download) |
| `ModelName` | *(required)* | HuggingFace model identifier for small tier |
| `QuantizationMethod` | `awq` | Quantization method (`awq`, `gptq`, `none`) |
| `MaxSequenceLength` | `2048` | Max sequence length for small model |
| `MinTaskCount` | `0` | Min small-tier tasks (0 = scale-to-zero) |
| `MaxTaskCount` | `10` | Max small-tier tasks |
| `LargeModelS3Path` | `none` | S3 URI for large model |
| `LargeModelName` | *(required)* | HuggingFace model identifier for large tier |
| `LargeModelQuantization` | `awq` | Quantization for large model |
| `LargeModelMaxSequenceLength` | `4096` | Max sequence length for large model |
| `LargeModelMinTaskCount` | `0` | Min large-tier tasks |
| `LargeModelMaxTaskCount` | `4` | Max large-tier tasks |
| `LargeModelGPUCount` | `8` | GPUs per large-tier task |
| `PromptComplexityThreshold` | `2000` | Prompt char length threshold for large routing |
| `MaxTokensComplexityThreshold` | `1024` | maxTokens threshold for large routing |
| `QueueDepthScaleThreshold` | `5` | Queue depth for scaling metric |
| `ScaleDownCooldownSeconds` | `300` | Cooldown before scaling down |
| `LogRetentionDays` | `30` | CloudWatch log retention |
| `OutputDestination` | `""` | S3 URI or SQS URL for results |
| `CostAllocationTagProject` | `gpu-inference-pipeline` | Cost allocation tag |

## GPU Auto-Repair

ECS Managed Instances use NVIDIA Data Center GPU Manager (DCGM) to continuously monitor GPU health. Critical XID errors trigger automatic instance replacement:

| XID | Description |
|---|---|
| 48 | Double Bit ECC Error |
| 74 | NVLink Error |
| 79 | GPU has fallen off the bus |
| 95 | Uncontained memory error |
| 140 | Unrecoverable ECC Error |

The auto-repair workflow follows a start-before-stop pattern:
1. ECS sets the impaired instance to DRAINING, blocking new task placements
2. Capacity provider provisions a healthy replacement instance
3. Existing tasks are allowed to stop gracefully (respecting `stopTimeout`)
4. Once the replacement is serving, the impaired instance is terminated

Rate limit: at most 20% of instances in a capacity provider (minimum 1) can be drained simultaneously.

## GPU Metrics and Observability

Metrics are published to CloudWatch Container Insights automatically, no agent installation or sidecar required:

- `TaskGPUUtilization` compute utilization percentage
- `TaskGPUMemoryUtilization` VRAM utilization percentage
- `TaskGPUTemperature` temperature in Celsius
- `TaskGPUPowerDraw` power consumption in watts
- `TaskGPURestartAppXidCount` accumulated XID error count
- `InstanceGPUUsageTotal` / `InstanceGPULimit` fleet capacity vs allocation

Three CloudWatch Alarms protect the pipeline:
- GPU temperature > 90°C → SNS notification
- XID error count > 0 → SNS notification (early warning before auto-repair)
- DLQ depth > 0 → SNS notification (processing failures)

## Infrastructure Provisioned (~70 resources)

- **Networking**  VPC, public/private subnets (2 AZs), NAT Gateway, VPC Endpoints (S3, SQS, ECR, CloudWatch Logs)
- **Ingestion**  API Gateway HTTP API (`POST /infer`), Router Lambda with SQS + CloudWatch + Bedrock permissions
- **Queues**  Small request queue (300s visibility), large request queue (600s visibility), shared DLQ (14-day retention)
- **Compute**  ECS cluster with 2 Managed Instances capacity providers:
  - Small: g6e.xlarge/2xlarge (1× L40S, 48 GB VRAM)
  - Large: g6e.48xlarge (8× L40S, 384 GB VRAM)
- **Services**  2 ECS services with independent desired counts and deployment circuit breakers
- **Container registry**  ECR repository with lifecycle policy (keep 5 tagged, expire untagged after 7 days)
- **IAM**  Task execution role, small task role, large task role, router role, infrastructure role, instance profile
- **Autoscaling** 2 Lambda-based composite metrics (1-min schedule), 2 sets of step scaling policies and alarms
- **Observability**  2 log groups, CloudWatch dashboard (4 sections), GPU temperature alarm, XID alarm, DLQ alarm, SNS topic

## Request / Response Format

**Request:**
```json
{
  "requestId": "uuid-v4",
  "prompt": "Your prompt text",
  "maxTokens": 256,
  "temperature": 0.7,
  "topP": 1.0,
  "route": "small"
}
```

Fields: `requestId` (required, UUID v4), `prompt` (required, non-empty string), `maxTokens` (optional, 1–2048), `temperature` (optional, 0.0–2.0), `topP` (optional, 0.0–1.0), `route` (optional, `"small"` or `"large"`).

**Response (synchronous from API Gateway):**
```json
{
  "requestId": "550e8400-e29b-41d4-a716-446655440000",
  "tier": "small"
}
```

**Result (asynchronous in S3):**
```json
{
  "requestId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "result": {
    "text": "Amazon ECS is a fully managed container orchestration service...",
    "usage": {"promptTokens": 17, "completionTokens": 102, "totalTokens": 119},
    "finishReason": "stop"
  },
  "processingTimeMs": 4956,
  "taskArn": "arn:aws:ecs:us-west-2:123456789012:task/cluster/task-id",
  "timestamp": "2026-05-03T04:35:47.906601+00:00"
}
```

## Running Tests

```bash
pip install -r tests/requirements.txt
pytest tests/ -v
```

## Key Design Decisions

- **Why asynchronous (SQS) instead of synchronous:** GPU inference is inherently variable-latency. A single request can take anywhere from 2 seconds to several minutes depending on prompt length and model size. A synchronous API would hold connections open for that entire window, making clients responsible for timeout handling and retry logic. SQS decouples producers from consumers: clients get an immediate acknowledgement and the worker processes at its own pace. Visibility timeout handles retries automatically if a task dies mid-flight. The DLQ captures poison messages without blocking the pipeline.

- **Why intelligent routing:** Running all requests through a 70B model wastes ~13× the compute cost on tasks a 7B model handles equally well. The Router Lambda adds a lightweight classification layer: prompt length, maxTokens, explicit override that routes each request to the right-sized model in under 10ms at effectively zero cost (Lambda free tier covers millions of invocations). Borderline cases include prompts of 500–2,000 chars, maxTokens ≤ 1,024, and no explicit override. For these, the router falls back to a fast Bedrock model (Nova Micro or Haiku) for semantic classification. In practice, this routing strategy saves 60–80% on compute costs compared to routing everything through the large tier.

- **Why ECS Managed Instances over Fargate or self-managed EC2:** Fargate doesn't support GPU workloads. Self-managed EC2 requires maintaining GPU-optimized AMIs with pre-configured NVIDIA drivers and Docker GPU runtimes. You also need to configure ECS agents and build custom health checks for hardware failures. With ECS Managed Instance, you get GPU access with Fargate-like operational simplicity: AMI management, ECS agent configuration, and GPU auto-repair are all handled by the service at baseline EC2 pricing with zero control-plane fees.

- **Why g6e over g5:** The g6e family with NVIDIA L40S GPUs provides 48 GB VRAM per GPU, versus 24 GB on the g5's A10G. That extra headroom accommodates larger KV caches for continuous batching, supports longer sequence lengths without truncation, and allows mid-size models (8B–13B) to run unquantized where g5 would require quantization or multi-GPU. The g6e.48xlarge's 8× L40S (384 GB aggregate VRAM) also makes it the natural fit for tensor-parallel 70B inference.

- **Why composite scaling metric:** Standard CPU-based autoscaling is blind to GPU workload demand. A single metric - queue depth or GPU utilization alone - leads to suboptimal scaling. Queue depth alone cannot detect a saturated GPU. Conversely, GPU utilization alone cannot anticipate demand before it arrives.
The composite metric uses the formula: max(queueDepth / threshold, gpuUtilization / 80)
•	Queue depth is a leading indicator that triggers scale-out before GPU memory saturates.
•	GPU utilization is a lagging indicator. It catches sustained saturation that the queue metric may miss during steady-state load.
A scheduled Lambda computes this per tier and publishes it as a custom CloudWatch metric.

- **Idempotency and at-least-once delivery:** SQS guarantees at-least-once delivery, meaning a message can be delivered more than once under failure conditions. The SQS worker checks S3 for an existing result (results/{requestId}.json) before invoking vLLM, and only deletes the message after successfully writing the result. This prevents duplicate inference runs without requiring a distributed lock or external state store.

- **Deployment circuit breaker:** GPU workloads have cold starts of 5-20 minutes (image pull + model download + engine initialization). Without a circuit breaker, a bad deployment (misconfigured environment variable, incompatible model format, OOM during load) cycles through failing tasks indefinitely, burning expensive GPU instance time.

- **Scale-to-Zero and Cold-Start Trade-offs:** Setting MinTaskCount=0 means no GPU instances run when queues are empty. This is the single largest cost lever, as a g6e.48xlarge left running costs approximately $320/day. However, scale-to-zero means cold starts when the first request arrives after an idle period.
Observed cold-start times:
•	Small tier (g6e.xlarge + Mistral-7B): ~7 minutes
•	Large tier (g6e.48xlarge + Llama-2-70B): ~15-20 minutes
Mitigations:
- *Keep MinTaskCount=1* for latency-sensitive tiers. Cost: ~$24/day for g6e.xlarge, ~$320/day for g6e.48xlarge. Eliminates cold starts entirely.
- *Scheduled scaling*: Keep instances warm during expected traffic hours and scale to zero during off-hours. For example, configure the small tier to maintain MinTaskCount=1 Monday through Friday 8am-6pm, and scale to zero on evenings and weekends. This reduces warm time from 168 hours/week to approximately 50 hours/week (roughly 70% savings compared to always-warm) while avoiding cold starts during business hours.
- *AWQ quantization* reduces model weight size (Mistral-7B from ~14 GB to 3.9 GB), directly reducing the time to load weights into VRAM.
- *Pre-bake small models into the container image* For models under 10 GB (like Mistral-7B-AWQ at 3.9 GB), including weights in the image eliminates the runtime download step. The trade-off is a larger image to pull.
- *Parallel S3 download* for larger models. aws s3 cp with multipart transfer maximizes network throughput.
- *SQS visibility timeout as a buffer* Set visibility timeout to exceed your cold-start time (300s for small tier, 600s for large tier). Messages remain invisible while the instance provisions and the model loads, then become available for processing without returning to the queue prematurely.

## Cost Considerations

| | Small Tier | Large Tier |
|---|---|---|
| Instance | g6e.xlarge (~$1.01/hr) | g6e.48xlarge (~$13.35/hr) |
| GPUs | 1× L40S (48 GB) | 8× L40S (384 GB) |
| Scale-in cooldown | 300s | 600s |
| Scale-in eval periods | 5 (5 min) | 10 (10 min) |

- **Scale-to-zero** on both tiers — no GPU costs when queues are empty
- **Routing saves 60–80%** — most requests handled by the cheap small tier
- **Zero control-plane fees** — unlike EKS ($0.10/hr per cluster)
- **Compute Savings Plans** — up to 37% reduction for steady-state workloads
- **ECR lifecycle policies** — auto-expire old images (~8 GB each)

## Cleanup

To avoid ongoing charges, remove all resources created during this walkthrough:

```bash
# Empty and delete S3 buckets
aws s3 rm s3://${STACK_NAME}-results-${AWS_REGION} --recursive
aws s3 rb s3://${STACK_NAME}-results-${AWS_REGION}
aws s3 rb s3://${STACK_NAME}-cfn-${AWS_REGION} --force

# Delete EventBridge rule (if created)
aws events remove-targets --rule ecs-gpu-instance-health --ids 1 --region $AWS_REGION 2>/dev/null
aws events delete-rule --name ecs-gpu-instance-health --region $AWS_REGION 2>/dev/null

# Delete the stack
aws cloudformation delete-stack --stack-name $STACK_NAME --region $AWS_REGION
aws cloudformation wait stack-delete-complete --stack-name $STACK_NAME --region $AWS_REGION
```

ECS Managed Instances automatically terminates backing instances when the capacity provider is deleted. Verify in the EC2 console that no orphaned instances remain after the stack deletion completes.

## Security

- All ECS tasks run in private subnets with no public IPs
- VPC endpoints keep traffic to S3, SQS, ECR, and CloudWatch off the public internet
- Each task role is scoped to least-privilege (specific buckets, queues, log groups)
- ECS Managed Instances restricts SSH and SSM access by default
- Images are scanned on push for CVEs
- All data at rest (S3 SSE, SQS SSE) and in transit (TLS) is encrypted

## License

See [LICENSE](LICENSE) for details.
