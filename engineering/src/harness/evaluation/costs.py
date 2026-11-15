"""计划和独立评价共用的费用口径。"""
from math import fsum


def operation_costs(operations):
    """按操作汇总实际费用，同一工作请求只计一次，缺测的时间和服务用量保持缺测。"""
    times = [op.get("elapsed_seconds") for op in operations]
    measured = [value for value in times if value is not None]
    # 同一不可变模型/输入的工作结果可供多个操作复用，CPU按任务身份计一次。
    workers = list({op["execution"].get("request_id",str(i)):op["execution"]
                    for i,op in enumerate(operations) if op.get("execution")}.values())
    calls = [call for op in operations for call in op.get("service_calls", [])]
    real = [c for c in calls if c["evidence_kind"] == "service_response"]
    simulated = [c for c in calls if c["evidence_kind"] == "simulated_service"]
    def tokens(group):
        return sum(c["usage"]["total_tokens"] for c in group) if group and all(c["usage"] is not None for c in group) else None
    return {"demo_units": sum(op["demo_units"] for op in operations),
            "measured_elapsed_seconds": fsum(measured),
            "unmeasured_operations": len(times) - len(measured),
            "operation_count": len(operations),
            "failed_operations": sum(op["status"] == "failed" for op in operations),
            "worker_cpu_seconds": fsum(w["cpu_seconds"] for w in workers) if workers and all(w.get("cpu_seconds") is not None for w in workers) else None,
            "observed_peak_worker_rss_bytes": max(w["observed_peak_rss_bytes"] for w in workers) if workers and all(w.get("observed_peak_rss_bytes") is not None for w in workers) else None,
            "unmeasured_workers":sum(w.get("cpu_seconds") is None or w.get("observed_peak_rss_bytes") is None for w in workers),
            "worker_count": len(workers), "real_service_tokens": tokens(real),
            "simulated_service_tokens": tokens(simulated),
            "service_calls": len(calls), "unknown_usage_calls": sum(c["usage"] is None for c in calls)}
