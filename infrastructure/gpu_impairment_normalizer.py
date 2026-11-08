import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)
ecs = boto3.client("ecs")
VERSION = "1.0.0"
XIDS = {46, 48, 54, 62, 64, 74, 79, 95, 109, 110, 136, 140, 142, 143, 151, 155, 156, 158}
XID_RE = re.compile(r"^XID_(\d+)$")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?")


def ts(v):
    if isinstance(v, datetime):
        v = v.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    m = TS_RE.match(v or "")
    if not m:
        return None
    return "%s.%sZ" % (m.group(1), (m.group(2) or "0")[:3].ljust(3, "0"))


def build_record(event, tasks, ci, now):
    """Contract gpu_impairment record, or None if not an ACCELERATED_COMPUTE impairment."""
    d = event.get("detail", {})
    hc = next((h for h in d.get("healthChecks", []) if h.get("type") == "ACCELERATED_COMPUTE"
               and h.get("status") == "IMPAIRED"), None)
    if hc is None:
        return None
    reason = hc.get("statusReason") or "UNSPECIFIED"
    m = XID_RE.match(reason)
    a = {
        "gpu.impairment.cause": "other",
        "gpu.health_check.type": "ACCELERATED_COMPUTE",
        "gpu.health_check.status": "IMPAIRED",
        "gpu.health_check.status_reason": reason,
        "gpu.health_check.last_status_change": ts(hc.get("lastStatusChange")) or ts(event.get("time")),
        "forensics.captured_at": ts(now),
        "source.event_id": event.get("id") or "unknown",
    }
    if tasks is not None:  # None = capture failed: leave the list out (contract: not captured)
        a["forensics.task_arns"] = sorted({t["taskArn"] for t in tasks})
        fams = sorted({t["taskDefinitionArn"].split("/")[-1].split(":")[0]
                       for t in tasks if t.get("taskDefinitionArn")})
        if fams:
            a["forensics.task_families"] = fams
    if m and int(m.group(1)) in XIDS:
        a["gpu.impairment.cause"] = "xid"
        a["gpu.xid.code"] = int(m.group(1))
    elif m:
        log.warning("XID %s is not in the contract enum; recording cause=other", m.group(1))
    at = {x.get("name"): x.get("value") for x in ci.get("attributes", [])}
    r = {
        "service.name": os.environ.get("SERVICE_NAME", "gpu-impairment-normalizer"),
        "cloud.region": event.get("region"),
        "cloud.availability_zone": at.get("ecs.availability-zone"),
        "aws.ecs.cluster.arn": d.get("clusterArn"),
        "aws.ecs.container_instance.arn": d.get("containerInstanceArn"),
        "aws.ecs.capacity_provider.name": d.get("capacityProviderName") or ci.get("capacityProviderName"),
        "host.id": d.get("ec2InstanceId") or ci.get("ec2InstanceId"),
        "host.type": at.get("ecs.instance-type"),
    }
    return {
        "contract_version": VERSION,
        "event_name": "gpu_inference.gpu_impairment",
        "event_id": str(uuid.uuid4()),
        "timestamp": ts(datetime.now(timezone.utc)),
        "resource": {k: v for k, v in r.items() if v},
        "attributes": a,
    }


def handler(event, context):
    d = event.get("detail", {})
    cluster, ci_arn = d.get("clusterArn"), d.get("containerInstanceArn")
    tasks, ci = None, {}
    try:  # forensic capture: tasks still on the instance before auto-repair drains it
        arns = ecs.list_tasks(cluster=cluster, containerInstance=ci_arn)["taskArns"]
        tasks = ecs.describe_tasks(cluster=cluster, tasks=arns[:100])["tasks"] if arns else []
    except Exception as e:
        log.warning("forensic task capture failed: %s", e)
    try:
        cis = ecs.describe_container_instances(cluster=cluster, containerInstances=[ci_arn])
        ci = (cis.get("containerInstances") or [{}])[0]
    except Exception as e:
        log.warning("container instance lookup failed: %s", e)
    rec = build_record(event, tasks, ci, datetime.now(timezone.utc))
    if rec:
        print(json.dumps(rec, separators=(",", ":"), sort_keys=True))
    return {"emitted": bool(rec)}
