
import io, os
import json
from google.cloud import storage
import pandas as pd
import pickle
import unicodedata
import string
from io import StringIO
import re

def gera_conjuntos(df, bucket):
    """
    Function to separate the data between those that will be used for model testing, validation and training
    """
    X, X_test, Y, y_test = train_test_split(df.drop(
        'Tag', axis=1), df['Tag'], test_size=0.2, stratify=df[['Tag']], random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=42)

    X_train = pd.concat((X_train, y_train), axis=1)
    X_train['ML_USE'] = 'TRAIN'
    X_test = pd.concat((X_test, y_test), axis=1)
    X_test['ML_USE'] = 'TEST'
    X_val = pd.concat((X_val, y_val), axis=1)
    X_val['ML_USE'] = 'VALIDATE'

    df_modelo = pd.concat((X_train, X_test, X_val))

    return df_modelo

def export_gcs(export_object, output_path, bucket):
    """
    Export export_object to output_path with a connected bucket.
    Usage:
    export_object: object file that will be exported
    output_path: output path inside bucket
    bucket: connected bucket via get_bucket
    """

    try:
        export_object = json.dumps(export_object)
    except:
        pass
    
    if isinstance(export_object, (pd.DataFrame, pd.Series, str)):
        f = io.StringIO()
        
        if isinstance(export_object, (pd.DataFrame, pd.Series)):
            export_object.to_csv(f, index=False, header=True)
        
        elif isinstance(export_object, str):
            f.write(export_object)
    else:
        f = io.BytesIO()
        pickle.dump(export_object, f)
    
    f.seek(0)

    blob = bucket.blob(output_path)

    # TODO: criar condição que verifica o tamanho do arquivo e reduz o chunksize
    # apenas caso necessário.
    ## For slow upload speed
    storage.blob._DEFAULT_CHUNKSIZE = 1024 * 1024 * 20# = 10 MB
    #storage.blob._MAX_MULTIPART_SIZE = 1024 * 1024 * 2

    blob.upload_from_file(f,  timeout=9999)


def import_data(project_id, compute_region, dataset_display_name, path, client):
    """Import structured data."""

    from google.cloud import automl_v1beta1 as automl

    response = None
    if path.startswith("bq"):
        response = client.import_data(
            dataset_display_name=dataset_display_name, bigquery_input_uri=path
        )
    else:
        # Get the multiple Google Cloud Storage URIs.
        input_uris = path.split(",")
        response = client.import_data(
            dataset_display_name=dataset_display_name,
            gcs_input_uris=input_uris
        )

    print("Processing import...")
    # synchronous check of operation status.
    print("Data imported. {}".format(response.result()))

    # [END automl_tables_import_data]


def create_dataset(project_id, compute_region, dataset_display_name, client):
    """Create a dataset."""

    from google.cloud import automl_v1beta1 as automl

    # Create a dataset with the given display name
    dataset = client.create_dataset(dataset_display_name)

    # Display the dataset information.
    print("Dataset name: {}".format(dataset.name))
    print("Dataset id: {}".format(dataset.name.split("/")[-1]))
    print("Dataset display name: {}".format(dataset.display_name))
    print("Dataset metadata:")
    print("\t{}".format(dataset.tables_dataset_metadata))
    print("Dataset example count: {}".format(dataset.example_count))
    print("Dataset create time:")
    print("\tseconds: {}".format(dataset.create_time.seconds))
    print("\tnanos: {}".format(dataset.create_time.nanos))

    # [END automl_tables_create_dataset]

    return dataset


def get_dataset(project_id, compute_region, dataset_display_name, client):
    """Get the dataset."""

    from google.cloud import automl_v1beta1 as automl

    # Get complete detail of the dataset.
    dataset = client.get_dataset(dataset_display_name=dataset_display_name)

    # Display the dataset information.
    print("Dataset name: {}".format(dataset.name))
    print("Dataset id: {}".format(dataset.name.split("/")[-1]))
    print("Dataset display name: {}".format(dataset.display_name))
    print("Dataset metadata:")
    print("\t{}".format(dataset.tables_dataset_metadata))
    print("Dataset example count: {}".format(dataset.example_count))
    print("Dataset create time:")
    print("\tseconds: {}".format(dataset.create_time.seconds))
    print("\tnanos: {}".format(dataset.create_time.nanos))

    # [END automl_tables_get_dataset]

    return dataset


def get_table_spec(project_id, compute_region, dataset_id, table_spec_id, client):
    """Get the table spec."""

    from google.cloud import automl_v1beta1 as automl

    # Get the full path of the table spec.
    table_spec_name = client.auto_ml_client.table_spec_path(
        project_id, compute_region, dataset_id, table_spec_id
    )

    # Get complete detail of the table spec.
    table_spec = client.get_table_spec(table_spec_name)

    # Display the table spec information.
    print("Table spec name: {}".format(table_spec.name))
    print("Table spec id: {}".format(table_spec.name.split("/")[-1]))
    print(
        "Table spec time column spec id: {}".format(
            table_spec.time_column_spec_id
        )
    )
    print("Table spec row count: {}".format(table_spec.row_count))
    print("Table spec column count: {}".format(table_spec.column_count))

    # [END automl_tables_get_table_spec]


def list_table_specs(project_id, compute_region, dataset_display_name, filter_, client):
    """List all table specs."""
    result = []

    from google.cloud import automl_v1beta1 as automl

    # List all the table specs in the dataset by applying filter.
    response = client.list_table_specs(dataset_display_name=dataset_display_name, filter_=filter_)

    print("List of table specs:")
    for table_spec in response:
        # Display the table_spec information.
        print("Table spec name: {}".format(table_spec.name))
        print("Table spec id: {}".format(table_spec.name.split("/")[-1]))
        print(
            "Table spec time column spec id: {}".format(
                table_spec.time_column_spec_id
            )
        )
        print("Table spec row count: {}".format(table_spec.row_count))
        print("Table spec column count: {}".format(table_spec.column_count))

        # [END automl_tables_list_specs]
        result.append(table_spec)

    return result


def list_column_specs(project_id, compute_region, dataset_display_name, filter_, client):
    """List all column specs."""
    result = []

    from google.cloud import automl_v1beta1 as automl

    # List all the table specs in the dataset by applying filter.
    response = client.list_column_specs(dataset_display_name=dataset_display_name, filter_=filter_)

    print("List of column specs:")
    for column_spec in response:
        # Display the column_spec information.
        print("Column spec name: {}".format(column_spec.name))
        print("Column spec id: {}".format(column_spec.name.split("/")[-1]))
        print("Column spec display name: {}".format(column_spec.display_name))
        print("Column spec data type: {}".format(column_spec.data_type))

        # [END automl_tables_list_column_specs]
        result.append(column_spec)

    return result


def get_column_spec(project_id, compute_region, dataset_id, table_spec_id, column_spec_id, client):
    """Get the column spec."""

    from google.cloud import automl_v1beta1 as automl

    # Get the full path of the column spec.
    column_spec_name = client.auto_ml_client.column_spec_path(
        project_id, compute_region, dataset_id, table_spec_id, column_spec_id
    )

    # Get complete detail of the column spec.
    column_spec = client.get_column_spec(column_spec_name)

    # Display the column spec information.
    print("Column spec name: {}".format(column_spec.name))
    print("Column spec id: {}".format(column_spec.name.split("/")[-1]))
    print("Column spec display name: {}".format(column_spec.display_name))
    print("Column spec data type: {}".format(column_spec.data_type))
    print("Column spec data stats: {}".format(column_spec.data_stats))
    print("Column spec top correlated columns\n")
    for column_correlation in column_spec.top_correlated_columns:
        print(column_correlation)

    # [END automl_tables_get_column_spec]


def update_dataset(project_id,compute_region,dataset_display_name,target_column_spec_name,weight_column_spec_name,test_train_column_spec_name,client):
    """Update dataset."""

    from google.cloud import automl_v1beta1 as automl

    if target_column_spec_name is not None:
        response = client.set_target_column(
            dataset_display_name=dataset_display_name,
            column_spec_display_name=target_column_spec_name,
        )
        print("Target column updated. {}".format(response))
    if weight_column_spec_name is not None:
        response = client.set_weight_column(
            dataset_display_name=dataset_display_name,
            column_spec_display_name=weight_column_spec_name,
        )
        print("Weight column updated. {}".format(response))
    if test_train_column_spec_name is not None:
        response = client.set_test_train_column(
            dataset_display_name=dataset_display_name,
            column_spec_display_name=test_train_column_spec_name,
        )
        print("Test/train column updated. {}".format(response))

    # [END automl_tables_update_dataset]


def create_model(
    project_id,
    compute_region,
    dataset_display_name,
    model_display_name,
    train_budget_milli_node_hours,
    include_column_spec_names,
    exclude_column_spec_names,
    client
):
    """Create a model."""

    from google.cloud import automl_v1beta1 as automl

    # Create a model with the model metadata in the region.
    response = client.create_model(
        model_display_name,
        train_budget_milli_node_hours=train_budget_milli_node_hours,
        dataset_display_name=dataset_display_name,
        include_column_spec_names=include_column_spec_names,
        exclude_column_spec_names=exclude_column_spec_names,
    )

    print("Training model...")
    print("Training operation name: {}".format(response.operation.name))
    print("Training completed: {}".format(response.result()))

    # [END automl_tables_create_model]


def get_model_evaluation(
    project_id, compute_region, model_id, model_evaluation_id, client
):
    """Get model evaluation."""

    from google.cloud import automl_v1beta1 as automl

    # Get the full path of the model evaluation.
    model_evaluation_full_id = client.auto_ml_client.model_evaluation_path(
        project_id, compute_region, model_id, model_evaluation_id
    )

    # Get complete detail of the model evaluation.
    response = client.get_model_evaluation(
        model_evaluation_name=model_evaluation_full_id
    )

    print(response)
    # [END automl_tables_get_model_evaluation]
    return response


def get_model(project_id, compute_region, model_display_name, client):
    """Get model details."""

    from google.cloud import automl_v1beta1 as automl
    from google.cloud.automl_v1beta1 import enums

    # Get complete detail of the model.
    model = client.get_model(model_display_name=model_display_name)

    # Retrieve deployment state.
    if model.deployment_state == enums.Model.DeploymentState.DEPLOYED:
        deployment_state = "deployed"
    else:
        deployment_state = "undeployed"

    # get features of top importance
    feat_list = [
        (column.feature_importance, column.column_display_name)
        for column in model.tables_model_metadata.tables_model_column_info
    ]
    feat_list.sort(reverse=True)
    if len(feat_list) < 10:
        feat_to_show = len(feat_list)
    else:
        feat_to_show = 10

    # Display the model information.
    print("Model name: {}".format(model.name))
    print("Model id: {}".format(model.name.split("/")[-1]))
    print("Model display name: {}".format(model.display_name))
    print("Features of top importance:")
    for feat in feat_list[:feat_to_show]:
        print(feat)
    print("Model create time:")
    print("\tseconds: {}".format(model.create_time.seconds))
    print("\tnanos: {}".format(model.create_time.nanos))
    print("Model deployment state: {}".format(deployment_state))

    # [END automl_tables_get_model]

    return model


def display_evaluation(
    project_id, compute_region, model_display_name, filter_, client
):
    """Display evaluation."""

    from google.cloud import automl_v1beta1 as automl

    # List all the model evaluations in the model by applying filter.
    response = client.list_model_evaluations(
        model_display_name=model_display_name, filter_=filter_)

    classification_metrics = list(
        response)[1].classification_evaluation_metrics
    print(classification_metrics)
    if str(classification_metrics):
        confidence_metrics = classification_metrics.confidence_metrics_entry

        # Showing model score based on threshold of 0.5
        print("Model classification metrics (threshold at 0.5):")
        for confidence_metrics_entry in confidence_metrics:
            if confidence_metrics_entry.confidence_threshold == 0.5:
                print(
                    "Model Precision: {}%".format(
                        round(confidence_metrics_entry.precision * 100, 2)
                    )
                )
                print(
                    "Model Recall: {}%".format(
                        round(confidence_metrics_entry.recall * 100, 2)
                    )
                )
                print(
                    "Model F1 score: {}%".format(
                        round(confidence_metrics_entry.f1_score * 100, 2)
                    )
                )
        print("Model AUPRC: {}".format(classification_metrics.au_prc))
        print("Model AUROC: {}".format(classification_metrics.au_roc))
        print("Model log loss: {}".format(classification_metrics.log_loss))

    # [END automl_tables_display_evaluation]


def list_model_evaluations(
    project_id, compute_region, model_display_name, filter_, client
):
    """List model evaluations."""
    result = []

    from google.cloud import automl_v1beta1 as automl

    # List all the model evaluations in the model by applying filter.
    response = client.list_model_evaluations(
        model_display_name=model_display_name, filter_=filter_
    )

    print("List of model evaluations:")
    for evaluation in response:
        print("Model evaluation name: {}".format(evaluation.name))
        print("Model evaluation id: {}".format(evaluation.name.split("/")[-1]))
        print(
            "Model evaluation example count: {}".format(
                evaluation.evaluated_example_count
            )
        )
        print("Model evaluation time:")
        print("\tseconds: {}".format(evaluation.create_time.seconds))
        print("\tnanos: {}".format(evaluation.create_time.nanos))
        print("\n")
        # [END automl_tables_list_model_evaluations]
        result.append(evaluation)