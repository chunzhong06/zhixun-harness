"""按改动范围运行已有检查；默认保护数据、评价、版本和主要维护行为。"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]

# 核心范围以研究结果和基本可运行性为选择依据；完整联通和可选SDK按需核对。
CORE=(
    'engineering.tests.data.test_data_conditions',
    'engineering.tests.workflow.test_flow.FlowTests.test_adoption_labels_never_reach_planner_or_training',
    'engineering.tests.workflow.test_flow.FlowTests.test_only_one_challenger_gets_final_adoption_comparison',
    'engineering.tests.workflow.test_flow.FlowTests.test_low_budget_preserves_output_and_does_not_overdraw',
    'engineering.tests.workflow.test_flow.FlowTests.test_resume_after_adoption_decision_preserves_selected_bundle',
    'engineering.tests.workflow.test_flow.FlowTests.test_invalid_candidate_predictions_do_not_replace_incumbent',
    'engineering.tests.models.test_regression.RegressionTests.test_metrics_use_known_numeric_answers',
    'engineering.tests.models.test_regression.RegressionTests.test_mae_adoption_uses_lower_error_and_records_improvement',
    'engineering.tests.evaluation.test_reference_evaluation',
    'engineering.tests.models.test_training_tools.TrainingToolsTests.test_complete_pipeline_survives_reload_and_move',
    'engineering.tests.methods.test_maintenance.MaintenanceTests.test_plain_retraining_continues_adopted_recipe_and_recovers',
    'engineering.tests.methods.test_maintenance_mechanisms.MechanismTests.test_alarm_is_consumed_while_fresh_deterioration_still_triggers',
    'engineering.tests.methods.test_assessment.AssessmentTests.test_overall_gain_keeps_the_regressed_group_visible',
    'engineering.tests.methods.test_decision_loop.DecisionLoopTests.test_trial_then_inspection_of_candidate_then_changed_component',
    'engineering.tests.methods.test_decision_loop.DecisionLoopTests.test_untrainable_recent_pool_guides_history_replay_and_survives_recovery',
    'engineering.tests.runtime.test_runtime_budget.RuntimeBudgetTests.test_wall_budget_survives_wait_and_batch_change',
)
SCOPES={
    'core':CORE,
    'data':('engineering.tests.data.test_data_conditions','engineering.tests.data.test_handoff_splitting','engineering.tests.integrations.test_connections','engineering.tests.data.test_gas_input','engineering.tests.workflow.test_official_gas','engineering.tests.data.test_task_inputs'),
    'models':('engineering.tests.models.test_training_tools','engineering.tests.models.test_regression','engineering.tests.methods.test_maintenance.ComponentWiringTests','engineering.tests.models.test_initial_configuration'),
    'methods':('engineering.tests.methods.test_assessment','engineering.tests.methods.test_decision_loop','engineering.tests.methods.test_maintenance','engineering.tests.methods.test_maintenance_mechanisms','engineering.tests.methods.test_maintenance_completion','engineering.tests.methods.test_planning_facts','engineering.tests.methods.test_error_regions','engineering.tests.methods.test_action_selection','engineering.tests.methods.test_stage2_mechanisms'),
    'runtime':('engineering.tests.workflow.test_flow','engineering.tests.runtime.test_execution','engineering.tests.runtime.test_runtime_budget','engineering.tests.data.test_handoff_splitting','engineering.tests.runtime.test_storage_replacement','engineering.tests.workflow.test_supervised_assembly','engineering.tests.workflow.test_continuity'),
    'evaluation':('engineering.tests.evaluation.test_longitudinal','engineering.tests.evaluation.test_final_answers','engineering.tests.evaluation.test_reference_evaluation','engineering.tests.evaluation.test_comparison','engineering.tests.evaluation.test_shared_comparison','engineering.tests.evaluation.test_study_preparation'),
    'integrations':('engineering.tests.integrations.test_connections','engineering.tests.integrations.test_services','engineering.tests.integrations.test_offline_tracking'),
}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=(*SCOPES,'all'),default='core')
    parser.add_argument('--output',type=Path,help='需要留证据时保存日志和回执；默认不生成文件')
    args=parser.parse_args(argv)
    (ROOT / 'scratch').mkdir(exist_ok=True)
    loader=unittest.TestLoader()
    suite=(loader.discover(str(ROOT/'engineering/tests'), top_level_dir=str(ROOT)) if args.scope=='all' else loader.loadTestsFromNames(SCOPES[args.scope]))
    buffer=io.StringIO()
    # 终端以结果和失败原因为主，完整运行细节只在要求留存时写入指定目录。
    with contextlib.redirect_stdout(buffer),contextlib.redirect_stderr(buffer):
        result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
    report={'scope':args.scope,'passed':result.wasSuccessful(),'run':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),
            'skipped':[{'test':str(test),'reason':reason} for test,reason in result.skipped],
            'model_training':False,'real_model_called':False}
    if args.output:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'tests.log').write_text(buffer.getvalue(),encoding='utf-8')
        (args.output/'tests.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not result.wasSuccessful():
        print(buffer.getvalue());return 1
    print('所选范围检查通过。')
    for _,reason in result.skipped:print('未执行：'+reason)
    return 0


if __name__=='__main__':sys.exit(main())
