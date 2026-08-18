# GPU Auto-Repair Demo (XID Fault Injection)

Showcase Amazon ECS Managed Instances **GPU auto repair** in action by simulating an
NVIDIA XID error — no real hardware fault required. This is the ECS analogue of the
EKS [`xid-injection`](https://github.com/aws-samples/sample-eks-docs/tree/main/ai-ml/manifests/xid-injection)
sample: instead of injecting into a node monitoring agent from a `hostNetwork` pod, we
inject into the host's DCGM `nv-hostengine` from an ECS task and let the ECS agent's GPU health
monitor pick it up.

> **Verified end-to-end** on a live g6e.xlarge (NVIDIA L40S) Managed Instances cluster:
> injecting XID 79 marked the instance `IMPAIRED` (`XID_79`) in ~2 minutes, and auto repair
> drained it, launched a replacement (start-before-stop), and terminated the impaired instance
> — full cycle ~8 minutes. See [Verified result](#verified-result).

References:
- [GPU auto repair for Amazon ECS managed instances](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/managed-instances-gpu-auto-repair.html)
- [Monitor Amazon ECS container instance health](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/container-instance-health.html)
- [`AutoRepairConfiguration` API](https://docs.aws.amazon.com/AmazonECS/latest/APIReference/API_AutoRepairConfiguration.html)

> This demo builds on the stack deployed by the repo root README. It assumes the
> `${STACK_NAME}-cluster` cluster and its GPU Managed Instances capacity providers already
> exist and that at least one GPU container instance is running.

## Background: how ECS detects GPU faults and repairs instances

Amazon ECS uses the NVIDIA Data Center GPU Manager (**DCGM**) to monitor GPU health on
Managed Instances. When DCGM reports a critical GPU failure, the ECS agent surfaces it as an
`ACCELERATED_COMPUTE` container-instance health check whose `statusReason` carries the NVIDIA
XID code in the form `XID_<number>`, and marks the instance `IMPAIRED`.

With auto repair enabled (`autoRepairConfiguration.actionsStatus = ENABLED`), ECS replaces the
impaired instance using a **start-before-stop** workflow:

1. The impaired instance is set to `DRAINING` — no new tasks placed on it.
2. ECS provisions a replacement instance.
3. Existing tasks drain gracefully (the task stop timeout is honored).
4. After draining, ECS terminates the impaired instance.

ECS rate-limits repairs: at most 20% of a capacity provider's instances drain at once (or one
at a time when there are fewer than 9).

Detection path:

```
NVIDIA driver -> DCGM (field 230 / XID) -> ECS agent GPU health monitor
             -> ACCELERATED_COMPUTE = IMPAIRED (statusReason XID_<n>) -> auto repair
```

This stack already enables auto repair on **both** capacity providers in
`infrastructure/template.yaml`:

```yaml
ManagedInstancesProvider:
  AutoRepairConfiguration:
    ActionsStatus: ENABLED
```

### Monitored XID codes

ECS marks the instance impaired and replaces it for these XIDs (see the doc above for the full
table): `46, 48, 54, 62, 64, 74, 79, 95, 109, 110, 136, 140, 142, 143, 151, 155, 156, 158`.
`79` (GPU fell off the bus) is the default used here and lines up with the EKS sample.

## Injection method

DCGM exposes an error-injection framework. Writing a synthetic value into DCGM **field 230**
(the XID field) makes DCGM report that XID to every client watching that host engine —
including the ECS agent's health monitor — exactly as a real fault would.

On an ECS Managed Instance the host runs a single `nv-hostengine` (systemd unit `nvidia-dcgm`)
bound to a **unix domain socket at `/run/nvidia-dcgm/nv-hostengine`** — *not* TCP port 5555. The
ECS GPU health monitor reads that same engine, so the injector must reach it there:

```
dcgmi test --inject --gpuid 0 -f 230 -v 79 --host unix:///hostdcgm/nv-hostengine
```

On ECS there is no `kubectl`, so the injector is delivered as an ECS task:

- **`xid-inject-taskdef.json`** — a task definition using an NVIDIA DCGM image (ships `dcgmi`).
  Key details that make it work on Managed Instances:
  - It **bind-mounts the host directory `/run/nvidia-dcgm`** into the container (at
    `/hostdcgm`) so `dcgmi` can reach the host socket. `networkMode: host` / `pidMode: host`
    alone are **not** enough, because the container has its own mount namespace and the engine
    listens on a unix socket, not TCP.
  - It runs `privileged` and reserves **no GPU and no CPU** (`cpu: 0`). GPU instances here are
    already fully packed by the inference task, so requesting a GPU or CPU would leave the
    injector stuck `PENDING`. `dcgmi` talks to the host engine over the socket and does not need
    its own GPU reservation.
  - After injecting one XID it idles (`sleep 3600`) so it doubles as a live injection client via
    ECS Exec.
- **`run-injection.sh`** — registers the task definition and runs it via the GPU **capacity
  provider strategy** (Managed Instances tasks cannot use `--launch-type EC2`), then reports
  which instance it landed on.

One critical XID on one GPU is a complete test: `ACCELERATED_COMPUTE` is an instance-level
health check, so the first critical XID already marks the whole instance for replacement.

> **You cannot choose the instance.** Managed Instances placement-constraint expressions are
> limited to `ecs.subnet-id`, `ecs.vpc-id`, `ecs.availability-zone`, `ecs.cpu-architecture`, and
> `ecs.instance-type` — there is no `ec2InstanceId`. The injector lands on whichever GPU
> instance the capacity provider picks.

> DCGM version note: match the injector image's DCGM major version to the one the ECS Managed
> Instances AMI ships. The task definition defaults to `nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04`
> (DCGM 4.x); bump the tag if your instances run a different DCGM major version.

## Run it

```bash
export AWS_REGION=us-west-2
export STACK_NAME=ecs-gpu-inference

cd demos/gpu-auto-repair

# Inject XID 79 on GPU 0 of a GPU container instance (the capacity provider picks which).
./run-injection.sh 79 0
```

`run-injection.sh` resolves the stack's task/execution roles, grants the task role the
`ssmmessages` permissions ECS Exec needs, registers `xid-inject`, launches it via the GPU
capacity provider, and prints which instance it landed on plus the follow-up commands (log
tail, repair watch, on-demand exec).

Confirm the injection succeeded in the injector log — look for `Successfully injected field
info.` followed by `INJECT_OK`:

```bash
aws logs tail /ecs/xid-inject --follow --region $AWS_REGION
```

## Verify detection

Watch the container-instance health flip to `IMPAIRED` with the XID reason:

```bash
./observe-repair.sh
```

Expected output as the fault lands (abridged):

```
ec2                   state     overall    accel      reason    running
i-061278d6f2cb14435   ACTIVE    IMPAIRED   IMPAIRED   None      2
```

> Note: `DescribeContainerInstances` reports `ACCELERATED_COMPUTE = IMPAIRED` but often leaves
> `statusReason` empty. The XID code (`XID_79`) reliably shows up in the **EventBridge health
> event** (see below), which is why the demo also captures those events.

Equivalent one-shot API check:

```bash
aws ecs describe-container-instances \
  --cluster ${STACK_NAME}-cluster \
  --container-instances <container-instance-arn> \
  --include CONTAINER_INSTANCE_HEALTH \
  --region $AWS_REGION \
  --query 'containerInstances[0].healthStatus'
```

### Health-change events (ships with the stack)

The CloudFormation template already provisions an EventBridge rule
(`${STACK_NAME}-container-instance-health`) that routes every `ECS Container Instance Health
Change` event where the instance becomes `IMPAIRED` to a CloudWatch Logs group **and** the
stack's SNS alarm topic. So the fault and the repair activity are recorded automatically:

```bash
aws logs tail /ecs/${STACK_NAME}/container-instance-health --follow --region $AWS_REGION
```

> If you are running against a cluster **not** created by this template, use the standalone
> `setup-health-events.sh` (and `health-events-rule.json`) instead — it creates an equivalent
> rule + log group on its own.

## Observe the full repair cycle

After the critical XID marks the instance `IMPAIRED`, auto repair drains and replaces it. Keep
`observe-repair.sh` running and you should see:

```
OK  ->  IMPAIRED (XID_79)  ->  DRAINING  ->  (instance terminated)
                              a replacement instance registers and goes OK
```

During the `DRAINING` window the total instance count briefly rises (e.g. 10 → 11) because ECS
provisions the replacement **before** terminating the impaired instance (start-before-stop),
then settles back once the old instance is gone.

## Verified result

Run against a live `gpu-inference-ecs-gpu-inference-capacity` provider (g6e.xlarge, 1× NVIDIA
L40S, 10 instances), injecting `XID 79`:

| Time (UTC) | Event | Fleet size |
|---|---|---|
| 06:06:15 | `dcgmi test --inject -f 230 -v 79` → `Successfully injected field info.` | 10 |
| 06:08:16 | ECS marks the instance `IMPAIRED` (`ACCELERATED_COMPUTE`) | 10 |
| 06:07:57 | EventBridge health event logged with `"statusReason":"XID_79"` | 10 |
| 06:09:25 | Instance → `DRAINING` (auto repair begins) | 10 |
| 06:10:19 | Replacement instance provisioned (start-before-stop) | 11 |
| 06:17:27 | Tasks fully drained (2 → 0) | 11 |
| 06:17:55 | Impaired instance terminated; fleet self-healed | 10 |

Detection ≈ 2 minutes after injection; full repair ≈ 8 minutes after `IMPAIRED`.

## On-demand injection (fire more XIDs without re-running)

The injector idles for an hour, so exec into it to inject any XID on demand. You must pass the
host socket via `--host` (there is no TCP listener):

```bash
aws ecs execute-command \
  --cluster ${STACK_NAME}-cluster \
  --task <injector-task-arn> \
  --container xid-inject \
  --interactive \
  --command "dcgmi test --inject --gpuid 0 -f 230 -v 74 --host unix:///hostdcgm/nv-hostengine" \
  --region $AWS_REGION
```

Use this to exercise different codes (for example `74` NVLink error, also a Replace). ECS Exec
requires the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
installed locally.

## Cleanup

```bash
# Stop the injector task if it is still idling
aws ecs stop-task --cluster ${STACK_NAME}-cluster --task <injector-task-arn> --region $AWS_REGION

# The EventBridge rule + health log group ship with the stack and are removed when
# you delete the CloudFormation stack. Delete the injector log group separately:
aws logs delete-log-group --log-group-name /ecs/xid-inject --region $AWS_REGION

# Only if you used the standalone setup-health-events.sh (non-stack cluster):
aws events remove-targets --rule ecs-gpu-health-impaired --ids cwlogs --region $AWS_REGION 2>/dev/null || true
aws events delete-rule --name ecs-gpu-health-impaired --region $AWS_REGION 2>/dev/null || true
aws logs delete-log-group --log-group-name /ecs/container-instance-health --region $AWS_REGION 2>/dev/null || true

# Remove the ECS Exec policy added to the task role
aws iam delete-role-policy --role-name ${STACK_NAME}-task-role --policy-name EcsExecForXidInject
```

## Caveats

- Injecting a critical XID **will** cause ECS to terminate and replace the instance when auto
  repair is enabled. Run it only against instances you are willing to lose. Because Managed
  Instances placement cannot target a specific instance, treat *any* GPU instance in the
  capacity provider as fair game.
- If `dcgmi` reports `Unable to connect to host engine`, check that (a) the host directory
  `/run/nvidia-dcgm` is bind-mounted into the container, (b) you pass `--host
  unix:///hostdcgm/nv-hostengine` (the engine uses a unix socket, not TCP 5555), and (c) the
  injector image's DCGM major version matches the AMI's.
- The injector reserves no CPU/GPU so it can co-locate on an already-packed GPU instance. If you
  run it on a cluster where GPU instances have spare capacity, this is still fine.
- An injected XID persists in DCGM until the instance is replaced; a `dcgmi` injection cannot be
  "un-injected". With auto repair enabled the faulted instance is replaced automatically, which
  clears the condition. If you disabled auto repair, terminate the instance manually to recover.
