COLUMNS_RENAMING = {
    "prob_desc_time_limit": "time_limit",
    "prob_desc_sample_outputs": "sample_outputs",
    "src_uid": "source_id",
    "prob_desc_notes": "problem_notes",
    "prob_desc_description": "problem_description",
    "prob_desc_output_spec": "output_specification",
    "prob_desc_input_spec": "input_specification",
    "prob_desc_output_to": "output_destination",
    "prob_desc_input_from": "input_source",
    "lang": "programming_language",
    "lang_cluster": "language_family",
    "difficulty": "difficulty_rating",
    "file_name": "file_name",
    "code_uid": "code_id",
    "prob_desc_memory_limit": "memory_limit",
    "prob_desc_sample_inputs": "sample_inputs",
    "exec_outcome": "execution_result",
    "source_code": "solution_code",
    "prob_desc_created_at": "creation_timestamp",
    "tags": "tags",
    "hidden_unit_tests": "hidden_tests"
    }
SELECTED_COLUMNS = [
    "time_limit",
    "sample_outputs",
    #"source_id", not actual prediction feature
    "problem_notes",
    "problem_description",
    "output_specification",
    "input_specification",
    #"output_destination",
    #"input_source",
    #"programming_language",
    #"language_family", Only one value
    "difficulty_rating",
    #"file_name", always same value
    #"code_id", not actual prediction feature
    "memory_limit",
    "sample_inputs",
    "execution_result",
    "solution_code",
    #"hidden_tests" always empty
    ]
SELECTED_TAGS = [
    'math', 
    'graphs', 
    'strings', 
    'number theory', 
    'trees', 
    'geometry', 
    'games', 
    'probabilities']

MINIMUM_TAGS = 0

COLUMNS_TO_LOWERCASE = [
    "problem_notes",
    "problem_description",
    "output_specification",
    "input_specification",
]

CREATION_TIME_COLUMN = "creation_timestamp"

TEST_SET_SIZE = 0.25

TEXT_FEATURE_COLUMNS = ['problem_description', 'problem_notes']
OTHER_FEATURE_COLUMNS = ['time_limit', 'memory_limit', 'difficulty_rating']
CODE_FEATURE_COLUMNS = ['solution_code']


PROBLEM_ID_COLUMN = 'problem_id'


FINE_TUNING_TEXT_COLUMN = "text" 

GROUND_TRUTH_TAG_COLUMNS = SELECTED_TAGS

PREDICTION_TAG_COLUMNS_PREFIX = "pred_"
PREDICTION_TAG_COLUMNS = [f"{PREDICTION_TAG_COLUMNS_PREFIX}{tag}" for tag in SELECTED_TAGS]