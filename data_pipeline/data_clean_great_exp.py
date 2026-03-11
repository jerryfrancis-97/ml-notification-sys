import great_expectations as gx 
import pandas as pd

def check_data_clean_quality():
    
    context = gx.get_context(context_root_dir="gx/")
    try:
        datasource = context.data_sources.get("my_datasource")
    except gx.exceptions.DataContextError:
        datasource = context.data_sources.add_pandas(name="my_datasource")
    data_asset_name = "training_data"
    batch_definition_name = "training_batch"

    try:
        # Try to fetch existing
        data_asset = datasource.get_asset(data_asset_name)
    except (LookupError, gx.exceptions.DataContextError):
        # If it doesn't exist, create it (e.g., as a Pandas DataFrame asset)
        data_asset = datasource.add_dataframe_asset(name=data_asset_name)

    try:
        # Try to fetch existing batch definition
        batch_definition = data_asset.get_batch_definition(batch_definition_name)
    except (LookupError, gx.exceptions.DataContextError):
        # If it doesn't exist, create it
        batch_definition = data_asset.add_batch_definition_whole_dataframe(
            batch_definition_name
        )


    path_to_data = "data/logs/training_data.csv"
    training_data = pd.read_csv(path_to_data)
    batch_parameters = {"dataframe": training_data}

    #getting data from the batch
    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    # expectation suite - data cleaning rules
    suite = gx.ExpectationSuite(name="data_cleaning_suite")
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id"))  
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="event_id"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="day"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="hour"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="day_of_week"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="is_weekend"))

    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="day_of_week", value_set=[0, 1, 2, 3, 4, 5, 6]))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="is_weekend", value_set=[0, 1]))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="hour", min_value=0, max_value=23))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="opened", value_set=[0, 1]))

    # Check if suite already exists in context; if so, use it, otherwise add new
    try:
        existing_suite = context.suites.get("data_cleaning_suite")
        suite = existing_suite
    except gx.exceptions.DataContextError:
        suite = context.suites.add(suite)


    #validation
    validation_definition_name = "my_validation_definition"
    validation_definition = gx.ValidationDefinition(
        data=batch_definition, suite=suite, name=validation_definition_name
    )
    try:
        validation_definition = context.validation_definitions.get("my_validation_definition")
    except gx.exceptions.DataContextError:
        validation_definition = context.validation_definitions.add(validation_definition)
    
    validation_result = validation_definition.run(batch_parameters=batch_parameters)
    # print(validation_result)

    #create a checkpoint
    try:
        checkpoint = context.checkpoints.get("my_checkpoint")
    except gx.exceptions.DataContextError:
        checkpoint = context.checkpoints.add(
            gx.Checkpoint(
                name="my_checkpoint",
                validation_definitions=[validation_definition]
            )
        )

    #do check
    checkpoint_result = checkpoint.run(batch_parameters=batch_parameters)

    if checkpoint_result.success:
        print("Validation passed successfully")
        checkpoint.save()
    else:
        print("Validation failed")
        print(checkpoint_result.get_validation_result().get_failure_cases())

    # context = context.convert_to_file_context()

if __name__=="__main__":
    check_data_clean_quality()
    print("Data cleaning quality check completed")