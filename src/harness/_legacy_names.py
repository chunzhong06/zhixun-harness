"""仅登记实际存在过的导入名。每组共用一个实现包，等号表示模块曾经改名。"""

LEGACY_MODULES = {}


def _names(package, target, names):
    entries = LEGACY_MODULES.setdefault(package, {})
    for item in names.split():
        name, _, changed = item.partition('=')
        entries[name] = target + '.' + (changed or name)


_names('harness.data', 'harness.adapters.datasets', '''
archive_replay bearing_input cmapss_input data_conditions development gas_input
official_bearing official_engine official_gas official_rul package_input selection splitting
''')

_names('harness.integrations', 'harness.adapters.platforms', '''
platform_train
''')

_names('harness.integrations', 'harness.adapters.providers', '''
pytrio_planner qwen_text_template research_broker service_research_broker service_transport
''')

_names('harness.integrations', 'harness.adapters.research', '''
author_comind author_compatibility author_executor author_mlestar author_recovery
author_task_runtime broad_transfer_task candidate_edit comind_branch_recovery
engine_parent_view feedback_task frontier_task generated_regressor native_author_task
native_research notebook_boundary official_component_driver official_driver
official_research_task offline_tracking proposal_inbox published_parent recipe_regressor
response_format tracking
''')

_names('harness.integrations', 'harness.adapters.tracking', '''
native_recording native_trace research_tracking
''')

_names('harness.integrations', 'harness.engine', '''
research_recovery service_budget
''')

_names('harness.integrations', 'harness.research', '''
candidate_selection free_research_driver research_ledger
''')

_names('harness.learning.models', 'harness.learning.models.air_quality', '''
aircubes aircubes_bias aircubes_changes aircubes_history aircubes_innovation aircubes_joint
aircubes_local_gain aircubes_normalize aircubes_pair aircubes_precision aircubes_response
aircubes_retrain aircubes_stack
''')

_names('harness.learning.models', 'harness.learning.models.flight', '''
flight_block flight_calibration_support flight_component flight_condition_response
flight_error_surface flight_fault_experts flight_forest_selection flight_health
flight_health_stage flight_horizons flight_inverse flight_load_neighbors flight_local_trials
flight_masked_readings flight_range_choice flight_route flight_segment_precision
flight_segment_training flight_sensor_trials flight_stage_risk flight_throttle
flight_wear_transfer
''')

_names('harness.learning.models', 'harness.learning.models.gas_field', '''
bottle_alignment bottle_challenge bottle_contrast dsps_cycle_release dsps_cycle_trials
dsps_dose_contrast dsps_response turbulent_contrast turbulent_reference turbulent_rescue
vapor_airflow vapor_boundary vapor_correction vapor_environment vapor_field vapor_reference
vapor_waveform
''')

_names('harness.learning.models', 'harness.learning.models.wind', '''
wind_donors wind_dynamic_head wind_history wind_limit wind_mapping wind_persistence wind_pmlp
wind_power wind_ramp_control wind_reference wind_regime wind_rotor wind_source_control
wind_torque wind_transition
''')

_names('harness.methods', 'harness.research.methods', '''
feedback_coverage feedback_query_risk feedback_sampling fleet_training flight_complement
flight_disagreement flight_life_pairs gas_audit_feedback gas_class_reference gas_phase_graph
gas_rank_readout gas_response_votes gas_returned_errors heater_disagreement training_inquiry
''')

_names('harness.methods.diagnosis', 'harness.research.methods.diagnosis', '''
error_regions inspection_followup investigation problem_tracking tool_registry
''')

_names('harness.methods.experience', 'harness.research.methods.experience', '''
method_guidance method_memory
''')

_names('harness.methods.planning', 'harness.research.methods.planning', '''
assessment baselines correction_planning decision_loop decisions error_slice gas_continuation
gas_proposal maintenance planning_facts rul_continuation
''')

_names('harness.methods.search', 'harness.research.methods.search', '''
action_selection cross_validation resource_allocation search_policy
''')

_names('harness.models', 'harness.core', '''
pipeline_recipe=recipe
''')

_names('harness.models', 'harness.learning.inference', '''
complete_checkpoint gas_views
''')

_names('harness.models', 'harness.learning.models', '''
agile_selector anchored_residual balanced_tree_residual bearing_landmarks bearing_load_clock
bearing_load_life bearing_phase_reference bearing_prefix bearing_prefix_release bearing_stumps
bearing_wear_rate binary_publication_bank calibration_query condition_processing
context_choice continuation_estimator continued_condition_engine corrections
coverage_scale_gas crossfit_parent_feature cumulative_classification_control cycle_install
delayed_teacher device_neighbors dynamic_gas_readout early_engine_reference engine_manifests
engine_setting_reference engine_source_choice engine_training_inquiry feedback_choice
feedback_neighbors feedback_routing feedback_training field_classification
field_driver_regression fleet_flight full_condition_engine gas_confirmed_choice
gas_feedback_bank gas_fit_queries gas_global_calibration gas_heater_archive
gas_heater_response gas_observed_queries gas_query_reuse gas_sampling gas_transitions
gas_vote_guard life_reference mixture_response model_artifacts observer_errors
official_gas_cpu official_rul_cpu online_aggregation paired_cycle_teacher paired_design
plume_training probability_fusion probability_threshold_classifier publication_control
publication_portfolio readout_training reference_numpy_mlp robust_residual_network
rul_selection scoped_calibration served_scale_gas settling_cycle sgp_cycle_network
site_training soft_state_classifier source_combinations sparse_reference station_learners
station_refresh station_residual stream_plan training_tools trajectory_training vote_feedback
weak_plume year_gas_calibration year_install_readout year_sensor_pairs
''')

_names('harness.models', 'harness.learning.models.air_quality', '''
aircubes aircubes_bias aircubes_changes aircubes_history aircubes_innovation aircubes_joint
aircubes_local_gain aircubes_normalize aircubes_pair aircubes_precision aircubes_response
aircubes_retrain aircubes_stack
''')

_names('harness.models', 'harness.learning.models.flight', '''
flight_block flight_calibration_support flight_component flight_condition_response
flight_error_surface flight_fault_experts flight_forest_selection flight_health
flight_health_stage flight_horizons flight_inverse flight_load_neighbors flight_local_trials
flight_masked_readings flight_range_choice flight_route flight_segment_precision
flight_segment_training flight_sensor_trials flight_stage_risk flight_throttle
flight_wear_transfer
''')

_names('harness.models', 'harness.learning.models.gas_field', '''
bottle_alignment bottle_challenge bottle_contrast dsps_cycle_release dsps_cycle_trials
dsps_dose_contrast dsps_response turbulent_contrast turbulent_reference turbulent_rescue
vapor_airflow vapor_boundary vapor_correction vapor_environment vapor_field vapor_reference
vapor_waveform
''')

_names('harness.models', 'harness.learning.models.wind', '''
wind_donors wind_dynamic_head wind_history wind_limit wind_mapping wind_persistence wind_pmlp
wind_power wind_ramp_control wind_reference wind_regime wind_rotor wind_source_control
wind_torque wind_transition
''')

_names('harness.models', 'harness.learning.training', '''
complete_training continuation_recipes native_continuation
''')

_names('harness.runtime', 'harness.adapters.storage', '''
storage=local
''')

_names('harness.adapters.platforms', 'harness.contest', '''
sdk_submission=submission sdk_gateway=gateway sdk_artifacts=artifacts lineage round_report
''')
_names('harness.adapters.platforms', 'harness.learning.training', 'shared_components=teacher_resources')

for package in ('harness.models', 'harness.learning.models'):
    _names(package, 'harness.learning.models.bearing', '''
    bearing_landmarks bearing_load_clock bearing_load_life bearing_phase_reference
    bearing_prefix bearing_prefix_release bearing_wear_rate
    ''')
