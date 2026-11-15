"""给官方工业工具绑定计划器。三种连接方式共用同一套动作检查与版本流程。"""
from harness.integrations.pytrio_planner import PyTrioConfig, PyTrioPlanner
from harness.integrations.service_transport import HttpChatTransport


def bind_driver(rule, folder, *, mode="rule", config=None, allow_network=False, api_key=None,
                simple_proposal=False, inbox=None, proposal_guidance=None):
    """真实服务须显式传入配置和授权；离线替身通过与服务相同的HTTP编解码边界。"""
    if mode == "rule" and not allow_network and config is None and not simple_proposal:
        return rule
    if mode == "local-proposal" and simple_proposal and inbox is not None and not allow_network and config is None:
        from harness.integrations.proposal_inbox import ProposalInbox
        config = PyTrioConfig("local-proposer", timeout_seconds=600)
        transport = ProposalInbox(inbox)
    elif mode == "mock" and not allow_network and config is None:
        from harness.testing.fixtures import SimulatedHttpService
        config = PyTrioConfig("offline-official-rehearsal")
        transport = HttpChatTransport(config, "fixture-only-key", wire=SimulatedHttpService(rule))
    elif mode == "service" and folder is not None and allow_network and isinstance(config, PyTrioConfig):
        if config.base_url.rstrip("/") != PyTrioConfig.base_url:
            raise ValueError("真实服务须使用已核对的 PyTRiO 官方地址")
        transport = HttpChatTransport(config, api_key, allow_network=True)
    else:
        raise ValueError("驱动方式、模型配置与网络开关不一致")
    audit = folder / "driver-exchanges.json" if folder is not None else None
    return PyTrioPlanner(config, transport, audit,
                         allow_network=allow_network, official_continuation=True, simple_proposal=simple_proposal,
                         proposal_guidance=proposal_guidance)
