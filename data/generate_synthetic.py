"""

A utility to generate a synthetic dataset of 500+ realistic Kubernetes incident post-mortems 
(OOMKilled, CrashLoopBackOff, DNS failures, etc....)

"""

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from typing import Any

# Ensure project root is on the path for cross-module imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

# Seed for reproducibility
random.seed(8304)
np.random.seed(8304)


# Incident Type Templates
# Each type has realistic descriptions, log patterns, and root causes


INCIDENT_TEMPLATES: list[dict[str, Any]] = [
    #OOMKilled
    {
        "type": "OOMKilled",
        "title_templates": [
            "{service} pod {pod} OOMKilled in namespace {namespace}", "Memory limit exceeded: {service} pod {pod} terminated",
            "Out of memory: {service} containers killed by OOM killer",
        ],
        "description_template": (
            "Pod {pod} in namespace {namespace} was terminated with reason OOMKilled. The container exceeded its memory limit of {memory_limit}. "
            "Last recorded memory usage was {memory_usage} ({usage_pct}% of limit). "
            "The pod has been restarted {restart_count} times in the last hour."
        ),
        "log_patterns": [
            "kernel: Out of memory: Kill process {pid} ({process_name})",
            "Memory cgroup out of memory: Killed process {pid} Container {container_id} was OOMKilled",
            "oom-kill:constraint=CONSTRAINT_MEMCG",
        ],
        "root_causes": [
            "Memory leak in application code - unbounded data structure growth",
            "Insufficient memory limits configured for workload. Sudden traffic spike causing increased memory allocation",
            "Third-party library memory leak under concurrent requests. Unoptimized database query loading entire result set into memory",
        ],
        "resolutions": [
            "Increased memory limit from {old_limit} to {new_limit}",
            "Fixed memory leak by closing database connections in finally block",
            "Added pod auto-scaling (HPA) with memory-based triggers",
            "Optimized query to use cursor-based pagination. Implemented circuit breaker to shed load during spikes",
        ],
        "services": ["payment-service", "order-service", "inventory-api", "user-auth"],
        "severities": ["Critical", "High", "High", "Medium"],
        "sev_weights": [0.4, 0.35, 0.15, 0.10],
    },
    
    #CrashLoopBackOff
    {
        "type": "CrashLoopBackOff",
        "title_templates": [
            "{service} pod {pod} in CrashLoopBackOff state",
            "Repeated crashes detected: {service} pod {pod} Container restart loop: {service} failing to start",
        ],
        "description_template": (
            "Pod {pod} in namespace {namespace} is in CrashLoopBackOff state. "
            "The container exits immediately after startup with exit code {exit_code} "
            "Error in container logs: '{error_message}'. The pod has restarted {restart_count} times in the last 30 minutes."
        ),
        "log_patterns": [
            "Error: {error_message}",
            "Fatal exception: {exception_type}",
            "Failed to start application: {error_detail}",
            "panic: runtime error: {go_error}",
        ],
        "root_causes": [
            "Misconfigured environment variable causing application failure",
            "Missing ConfigMap or Secret reference",  "Failed database connection on startup",
            "Invalid container entrypoint or command","Port conflict with host or sidecar container",
        ],
        "resolutions": [
            "Fixed environment variable configuration in ConfigMap",  "Created missing Secret '{secret_name}' for database credentials",
            "Added startup probe with 30s initial delay",   "Corrected container command to use proper entrypoint",
            "Resolved port conflict by changing container port to {new_port}",
        ],
        "services": ["payment-service", "order-service", "notification-svc", "auth-api", "api-gateway"],
        "severities": ["Critical", "Critical", "High", "High", "Medium"],
        "sev_weights": [0.5, 0.25, 0.15, 0.07, 0.03],
    },
    
    #ImagePullBackOff
    {
        "type": "ImagePullBackOff",
        "title_templates": [
            "{service} deployment failing: ImagePullBackOff for {image_name}",
            "Cannot pull container image: {service} pod {pod}",  "Image pull error: {image_name} not found or unauthorized",
        ],
        "description_template": (
            "Pod {pod} in namespace {namespace} is stuck in ImagePullBackOff. "
            "Failed to pull image '{image_name}' from registry {registry}. "  "Error: '{pull_error}'. The deployment was triggered {time_ago} minutes ago."
        ),
        "log_patterns": [
            "Failed to pull image \"{image_name}\": rpc error: code = {code}",
            "ErrImagePull: {pull_error_detail}", "Error response from daemon: pull access denied for {image_name}",
            "manifest for {image_name} not found",
        ],
        "root_causes": [
            "Incorrect image tag specified in deployment manifest",  "Registry authentication failure - missing imagePullSecret",
            "Docker registry rate limit exceeded",
            "Image does not exist in the specified registry",    "Network connectivity issue to container registry",
        ],
        "resolutions": [
            "Updated image tag from '{old_tag}' to '{new_tag}'",
            "Created imagePullSecret '{secret_name}' and referenced in ServiceAccount",
            "Switched to local registry mirror to avoid rate limits",  "Verified and corrected image path in deployment YAML",
            "Added registry credentials as Kubernetes secret",
        ],
        "services": ["payment-processor", "data-ingestion", "ml-inference", "cache-warmer"],
        "severities": ["High", "Medium", "Low"],"sev_weights": [0.5, 0.35, 0.15],
    },
    #Connection pool exhaustion
    {
        "type": "ConnectionPoolExhaustion",
        "title_templates": [
            "Database connection pool exhausted for {service}",  "{service} failing: PostgreSQL max_connections reached",
            "Connection timeout errors in {service} : pool starvation",
        ],
        "description_template": (
            "{service} is experiencing connection pool exhaustion. All {pool_size} connections in the pool are active. "
            "{waiting_count} requests are waiting for connections. Average wait time: {avg_wait}ms. "
            "Database {db_name} shows max_connections = {max_conn} reached."
        ),
        "log_patterns": [
            "FATAL: sorry, too many clients already",      "Timeout waiting for idle connection: pool exhausted",
            "Cannot acquire connection from pool within {timeout}s",   "Connection pool size {pool_size} exceeded for {db_name}",
        ],
        "root_causes": [
            "Connection leak - connections not returned to pool after use",
            "Sudden traffic spike to service causing pool saturation",     "Slow database queries causing connections to be held open",
            "Pool size misconfiguration - pool too small for workload", "Network latency between application and database",
        ],
        "resolutions": [
            "Increased connection pool size from {old_pool} to {new_pool}",
            "Fixed connection leak by ensuring connections closed in finally blocks",
            "Added PgBouncer connection pooler in front of PostgreSQL", "Optimized slow queries - 10x improvement in response time",
            "Implemented read replicas to distribute read load",
        ],
        "services": ["payment-service", "order-service", "user-service", "inventory-api"],
        "severities": ["Critical", "Critical", "High"],"sev_weights": [0.6, 0.3, 0.1],
    },
    #DNS Failures
    {
        "type": "DNSFailure",
        "title_templates": [
            "DNS resolution failures in {service} pod {pod}",   "Service discovery broken: {service} cannot resolve hostnames",
            "CoreDNS errors affecting pod {pod} in namespace {namespace}",
        ],
        "description_template": (
            "Pod {pod} in namespace {namespace} is experiencing DNS resolution failures. "
            "Unable to resolve hostname '{hostname}'. nslookup returns: '{dns_error}'. "
            "CoreDNS pods in kube-system are reporting {core_dns_status}."
        ),
        "log_patterns": [
            "lookup {hostname} on {dns_server}: no such host",  "dial tcp: lookup {hostname}: Temporary failure in name resolution",
            "getaddrinfo EAI_AGAIN {hostname}",  "plugin/errors: 2 {hostname}. A: read udp: i/o timeout",
        ],
        "root_causes": [
            "CoreDNS pods overloaded due to high query volume", "NetworkPolicy blocking DNS traffic (UDP port 53)",
            "CoreDNS ConfigMap misconfiguration",  "Node-level DNS cache corruption",
            "NDots configuration causing excessive DNS lookups",
        ],
        "resolutions": [
            "Scaled CoreDNS deployment to {replicas} replicas",   "Updated NetworkPolicy to allow UDP 53 traffic to kube-dns",
            "Fixed CoreDNS ConfigMap - added missing stub domain",
            "Flushed node DNS cache and restarted CoreDNS pods",  "Set ndots:2 in pod spec to reduce DNS query volume",
        ],
        "services": ["all-services", "api-gateway", "service-mesh"],
        "severities": ["Critical", "High", "High", "Medium"],  "sev_weights": [0.45, 0.30, 0.15, 0.10],
    },
    # CPU Throttling
    {
        "type": "CPUThrottling",
        "title_templates": [
            "CPU throttling detected for {service} pod {pod}", "{service} experiencing high latency due to CPU limits",
            "Container CPU throttled: {service} exceeding limits",
        ],
        "description_template": (
            "Pod {pod} in namespace {namespace} is being CPU throttled. "
            "CPU usage regularly exceeds limit of {cpu_limit}. Throttling rate: {throttle_pct}% over the last 5 minutes. "
            "P99 latency increased from {old_latency}ms to {new_latency}ms."
        ),
        "log_patterns": [
            "Container {container_id} is experiencing cpu throttling",
            "CPU usage {usage_pct}% exceeding threshold",  "Performance degradation due to CPU limit enforcement",
            "Warning: cpu.cfs_period_us throttling triggered",
        ],
        "root_causes": [
            "CPU limits set too low for actual workload",  "Inefficient code path consuming excessive CPU under load",
            "Missing HorizontalPodAutoscaler for variable workload",
            "Sidecar container consuming unexpected CPU resources",   "GC (garbage collection) overhead in JVM-based service",
        ],
        "resolutions": [
            "Increased CPU limit from {old_limit} to {new_limit} cores",
            "Optimized hot code path - 40% reduction in CPU usage", "Configured HPA with CPU utilization target of 70%",
            "Reduced sidecar CPU request and added resource quotas", "Tuned JVM GC settings (G1GC, MaxGCPauseMillis=200)",
        ],
        "services": ["payment-processor", "data-analytics", "ml-inference", "report-generator"],
        "severities": ["Medium", "Low", "Low"],"sev_weights": [0.5, 0.35, 0.15],
    },
    #network failures
    {
        "type": "NetworkFailure",
        "title_templates": [
            "Network connectivity loss in {service} pod {pod}", "Packet loss between {service} and {target_service}",
            "Network timeout: {service} cannot reach external API",
        ],
        "description_template": (
            "Pod {pod} in namespace {namespace} is experiencing network failures. Cannot connect to {target_service} at {target_host}:{target_port}. "
            "Error: '{network_error}'. Packet loss: {packet_loss}%. Latency: {latency}ms (normal: {normal_latency}ms)."
        ),
        "log_patterns": [
            "dial tcp {target_host}:{target_port}: i/o timeout", "connect: connection refused",
            "read tcp: connection reset by peer", "net/http: request canceled while waiting for connection",
        ],
        "root_causes": [
            "NetworkPolicy blocking egress traffic to required service",     "Service mesh (Istio/Linkerd) sidecar proxy misconfiguration",
            "Network partition between node groups", "Firewall rule blocking port {target_port}",
            "MTU mismatch causing packet fragmentation",
        ],
        "resolutions": [
            "Updated NetworkPolicy to allow egress to {target_host}:{target_port}",      "Restarted Istio sidecar proxy and verified Envoy configuration",
            "Resolved network partition by fixing VPC peering route tables",    "Added firewall rule allowing port {target_port} outbound",
            "Set MTU to 1400 in CNI configuration to match network path",
        ],
        "services": ["api-gateway", "payment-service", "notification-svc", "external-adapter"],
        "severities": ["Critical", "High", "Medium", "Medium"],    "sev_weights": [0.4, 0.35, 0.15, 0.10],
    },
]


# Supporting random data pools


NAMESPACES = ["production", "staging", "development", "monitoring", "kube-system"]
CLUSTERS = ["prod-us-east-1", "prod-eu-west-1", "staging-us-east", "dev-cluster"]

NODES = [f"node-{i:02d}" for i in range(1, 21)]
REGISTRIES = ["docker.io", "gcr.io", "ecr.aws", "ghcr.io", "quay.io"]

DB_NAMES = ["postgres-prod", "orders-db", "users-db", "analytics-db", "cache-db"]

EXIT_CODES = [1, 2, 126, 127, 128, 137, 139, 143]
GO_ERRORS = [
    "invalid memory address or nil pointer dereference",
    "index out of range", "send on closed channel",
    "concurrent map read and map write", "interface conversion",
]

EXCEPTION_TYPES = [
    "NullPointerException", "IllegalStateException",   "RuntimeException", "FileNotFoundException",
    "OutOfMemoryError", "StackOverflowError",
]


def generate_id(index: int) -> str:
    """Generate incident ID like INC-1045."""
    return f"INC-{1000 + index:05d}"


def generate_pod_name(service: str) -> str:
    """Generate a realistic pod name."""
    suffixes = ["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz1", "234"]

    return f"{service}-{random.randint(100, 999)}-{random.choice(suffixes)}"


def generate_timestamp(days_back: int = 90) -> str:
    """Generate a random timestamp within the last N days."""
    now = datetime.utcnow()
    delta = timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59),
    )
    return (now - delta).isoformat() + "Z"


def fill_template(template: str, **kwargs: Any) -> str:
    """Fill a template string with values, handling missing keys gracefully."""
    try:
        return template.format(**kwargs)
    except KeyError:
        # Fallback: just return template
        return template


def generate_incident(index: int) -> dict[str, Any]:
    """
    Generate a single synthetic Kubernetes incident with realistic data.
    """
    # Select incident type based on distribution
    template = random.choice(INCIDENT_TEMPLATES)
    incident_type = template["type"]

    title_tmpl = random.choice(template["title_templates"])
    severity = random.choices(template["severities"], weights=template["sev_weights"], k=1)[0]
    service = random.choice(template["services"])
    namespace = random.choice(NAMESPACES)
    pod = generate_pod_name(service)


    incident_id = generate_id(index)
    timestamp = generate_timestamp()

    # Generate realistic values
    memory_limit = f"{random.choice([128, 256, 512, 1024, 2048, 4096])}Mi"
    memory_usage = f"{random.choice([140, 280, 600, 1200, 2500, 4500])}Mi"

    usage_pct = random.randint(85, 200)
    restart_count = random.randint(3, 50)
    exit_code = random.choice(EXIT_CODES)
    error_msg = random.choice(GO_ERRORS) if incident_type == "CrashLoopBackOff" else \
        f"Connection refused to {random.choice(['postgres', 'redis', 'kafka', 'elasticsearch'])}"

    image_name = f"{random.choice(REGISTRIES)}/{service}:v{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}"
    pull_error = random.choice([
        "unauthorized: authentication required",  "manifest unknown: manifest not found",
        "dial tcp: i/o timeout",
        "toomanyrequests: rate limit exceeded","denied: requested access to the resource is denied",
    ])
    pool_size = random.choice([20, 50, 100, 200])
    db_name = random.choice(DB_NAMES)
    max_conn = pool_size + random.choice([0, 0, 0, 5, 10])
    cpu_limit = f"{random.choice([0.5, 1.0, 2.0, 4.0])}"
    old_latency_ms = random.choice([10, 25, 50, 100])
    new_latency_ms = old_latency_ms * random.choice([3, 5, 8, 15])

    # Build context for template filling
    ctx = {
        "service": service,"pod": pod, "namespace": namespace,  "memory_limit": memory_limit, "memory_usage": memory_usage,
        "usage_pct": usage_pct,   "restart_count": restart_count,
        "exit_code": exit_code, "error_message": error_msg,  "image_name": image_name,   "pull_error": pull_error,
        "registry": random.choice(REGISTRIES),  "time_ago": random.randint(5, 60),
        "pool_size": pool_size,  "db_name": db_name, "max_conn": max_conn,
        "waiting_count": random.randint(5, 50),     "avg_wait": random.randint(100, 5000),
        "cpu_limit": cpu_limit,    "throttle_pct": random.randint(10, 95),
        "old_latency": old_latency_ms,  "new_latency": new_latency_ms, "hostname": f"{service}.{namespace}.svc.cluster.local",
        "dns_error": random.choice(["server misbehaving", "no such host", "connection timed out"]),
        "target_service": random.choice(["redis-cache", "rabbitmq", "elasticsearch", "external-api"]),
        "target_host": f"10.0.{random.randint(0,255)}.{random.randint(1,254)}",
        "target_port": random.choice([5432, 6379, 5672, 9200, 9092]),

        "network_error": random.choice(["i/o timeout", "connection refused", "no route to host"]),
        "packet_loss": random.randint(5, 80),   "latency": random.randint(500, 5000),
        "normal_latency": random.randint(1, 50), "old_limit": random.choice(["128Mi", "256Mi", "0.5", "1.0"]),
        "new_limit": random.choice(["512Mi", "1024Mi", "2.0", "4.0"]),
        "old_pool": str(pool_size),  "new_pool": str(pool_size * random.choice([2, 3, 4, 5])),
        "old_tag": f"v{random.randint(1,3)}.{random.randint(0,3)}.{random.randint(0,9)}",  "new_tag": f"v{random.randint(4,5)}.{random.randint(0,5)}.{random.randint(0,9)}",
        "secret_name": f"{service}-registry-creds",  "replicas": str(random.choice([3, 5, 7, 10])),
        "pid": str(random.randint(1000, 99999)),   "process_name": random.choice(["java", "node", "python3", "gunicorn", "ruby"]),
        "container_id": f"containerd://{random.getrandbits(64):016x}",
        "code": str(random.choice(["Unknown", "NotFound", "Unauthorized", "DeadlineExceeded"])),
        "exception_type": random.choice(EXCEPTION_TYPES),
        "error_detail": random.choice(["config file not found", "cannot bind to port", "database unreachable"]),
        "go_error": random.choice(GO_ERRORS),"new_port": str(random.choice([8080, 8081, 8443, 9090])),
        "timeout": str(random.choice([5, 10, 30, 60])), "dns_server": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        "core_dns_status": random.choice(["high latency", "timeout errors", "NXD responses"]),
    }

    # Build incident
    title = fill_template(title_tmpl, **ctx)
    description = fill_template(template["description_template"], **ctx)
    root_cause = random.choice(template["root_causes"]).format(**ctx)
    resolution = random.choice(template["resolutions"]).format(**ctx)

    evidence_logs = [fill_template(log, **ctx) for log in random.sample(
        template["log_patterns"],
        k=min(3, len(template["log_patterns"]))
    )]

    incident: dict[str, Any] = {
        "incident_id": incident_id,
        "title": title,   "description": description,
        "incident_type": incident_type,
        "severity": severity, "root_cause": root_cause,
        "resolution": resolution,
        "evidence": {
            "logs": evidence_logs,
            "affected_pods": [pod, generate_pod_name(service)],
            "affected_services": [service] + random.sample(
                template["services"], k=min(2, len(template["services"]) - 1)
            ) if len(template["services"]) > 1 else [service],
            "cluster": random.choice(CLUSTERS),
            "node": random.choice(NODES),
        },
        "affected_services": list(set([service] + random.sample(
            template["services"],  k=min(2, len(template["services"]) - 1)
        ))),
        "affected_pods": [pod], "source": "synthetic",
        "timestamp": timestamp,
        "metadata": {
            "namespace": namespace, "cluster": random.choice(CLUSTERS),
            "node": random.choice(NODES),
        },
    }

    return incident


def generate_incidents(num_incidents: int = 500) -> list[dict[str, Any]]:
    """
    Generate a dataset of synthetic Kubernetes incidents
    """
    incidents = []
    type_counts: dict[str, int] = {}

    for i in range(num_incidents):
        incident = generate_incident(i)
        incidents.append(incident)
        incident_type = incident["incident_type"]
        type_counts[incident_type] = type_counts.get(incident_type, 0) + 1

    print(f"\n[OK] Generated {num_incidents} synthetic incidents")
    print("\nIncident distribution:")
    for incident_type, count in sorted(type_counts.items()):
        print(f"  {incident_type}: {count} ({count/num_incidents*100:.1f}%)")
    
    return incidents


def main():
    """Generate dataset and save to JSON."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic K8s incidents")

    parser.add_argument("--num-incidents", type=int, default=500, help="Number of incidents")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else \
        (Path(__file__).resolve().parent / "synthetic_incidents.json")

    incidents = generate_incidents(args.num_incidents)

    with open(output_path, "w") as f:
        json.dump(incidents, f, indent=2)

    print(f"\n[OK] Dataset saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    return incidents

#main method
if __name__ == "__main__":
    main()
