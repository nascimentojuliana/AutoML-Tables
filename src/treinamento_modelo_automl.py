#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import io
import re
import pandas as pd
import numpy as np
import unicodedata
import string
from io import StringIO

from google.cloud import automl_v1beta1 as automl

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='This script trains a model and makes predictions')
parser.add_argument('--project_id', type=str, help='GCP Project')
parser.add_argument('--compute_region', type=str, default=us-central1, help='compute region')
parser.add_argument('--input', type=str, default=us-central1, help='dataset for training the model, where the target is named Tag')
parser.add_argument('--bucket', type=str, help='Bucket used')
parser.add_argument('--model', type=str, help='model name')
parser.add_argument('--output_path', type=str, help='folder on storage for save the data used for training the model, ex: input/dataset.csv')
args = parser.parse_args()


if __name__ == "__main__":

    client = automl.TablesClient(project=args.project_id, region=args.compute_region)

    # Lendo os dados de entrada e nomeando como df
    df = pd.read_csv(args.input)

    #retira os casos duplicados no dataset de entrada
    df = df.drop_duplicates()

    # gera conjuntos que serão utilizados no processo (treinamento, teste e validação)
    utils.gera_conjuntos(df)

    #exporta conjunto para storage
    def export_gcs(export_object=df, output_path=args.output_path, bucket=args.bucket)

    # Cria dataset para modelo
    modelo = args.model
    dataset_display_name = '{modelo}'.format(modelo=modelo)
    utils.create_dataset(args.project_id, args.compute_region, dataset_display_name)

    # importa os dados no automl tables para dar inicio ao processo
    path = 'gs://{bucket}/{input}'.format(bucket=args.bucket, input=args.output_path)
    utils.import_data(project_id, compute_region, dataset_display_name, path)

    # Lista tabelas no dataset ids de tabela
    lista_tabela = utils.list_table_specs(
        project_id, compute_region, dataset_display_name, filter_=None)
    dataset_id = lista_tabela[0].name.split("/")[-3]
    table_spec_id = lista_tabela[0].name.split("/")[-1]

    utils.get_table_spec(args.project_id, args.compute_region, dataset_id, table_spec_id)

    # Lista ids de colunas
    lista_coluna = utils.list_column_specs(
        args.project_id, args.compute_region, dataset_display_name, filter_=None)
    for i in range(0, len(lista_coluna)):
        if lista_coluna[i].display_name == 'Tag':
            id_tag = lista_coluna[i].name.split("/")[-1]
        elif lista_coluna[i].display_name == 'ML_USE':
            id_ML_USE = lista_coluna[i].name.split("/")[-1]

    # Define as colunas de uso especial
    column_spec_id = id_tag
    utils.get_column_spec(args.project_id, args.compute_region, dataset_id,
                    table_spec_id, column_spec_id)
    column_spec_id = id_ML_USE
    utils.get_column_spec(args.project_id, args.compute_region, dataset_id,
                    table_spec_id, column_spec_id)

    target_column_spec_name = 'Tag'
    weight_column_spec_name = None
    test_train_column_spec_name = 'ML_USE'

    # Atualiza dataset
    utils.update_dataset(args.project_id, args.compute_region, dataset_display_name,
                   target_column_spec_name, weight_column_spec_name, test_train_column_spec_name)

    # Treina modelo
    model_display_name = '{modelo}'.format(modelo=modelo)
    train_budget_milli_node_hours = 1000
    utils.create_model(args.project_id, args.compute_region, dataset_display_name, model_display_name,
                 train_budget_milli_node_hours, include_column_spec_names=None, exclude_column_spec_names=None)

    # lista modelos treinados e avaliações
    utils.list_model_evaluations(args.project_id, args.compute_region,
                           model_display_name, filter_=None)
    utils.get_model(args.project_id, args.compute_region, model_display_name)
    utils.display_evaluation(args.project_id, args.compute_region,
                       model_display_name, filter_=None)