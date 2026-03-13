from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator  # correct
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'notification_data_pipeline',
    default_args=default_args,
    description='A pipeline to run simulator and dbt',
    schedule='@hourly',
    catchup=False
) as dag:

    run_simulator = BashOperator(
        task_id='run_simulator_and_logs',
        bash_command='python3 -m data_pipeline.simulator && python3 -m data_pipeline.generate_logs',
        cwd='/app',
    )

    run_dbt = BashOperator(
        task_id='run_dbt',
        bash_command='dbt run --profiles-dir .',
        cwd='/app/my_notification_dbt',
    )

    run_simulator >> run_dbt  # simulator runs first, then dbt