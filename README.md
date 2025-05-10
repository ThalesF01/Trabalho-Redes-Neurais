Sistema de Previsão de Operações Matemáticas com Redes Neurais
Este repositório contém um sistema de rede neural utilizando Keras, Optuna e técnicas avançadas de pré-processamento para prever operações matemáticas, como adição, subtração, multiplicação e divisão, com base em dois números fornecidos como entrada. O modelo usa otimização de hiperparâmetros com Optuna, validação cruzada, e callbacks avançados para monitoramento e melhoria do desempenho.

Estrutura do Repositório
O código está organizado em várias células (em formato Jupyter Notebook) e é composto pelos seguintes principais componentes:

Instalações e Imports

Métricas Customizadas e Callbacks

Geração e Divisão dos Dados

Pré-processamento de Dados

Criação do Modelo de Rede Neural

Otimização de Hiperparâmetros com Optuna

Treinamento e Validação

Exibição de MAES

Interface para Testes

Gráficos do treinamento


1. Instalações e Imports
Na primeira célula, são feitas as instalações das bibliotecas necessárias e importados os pacotes utilizados ao longo do código. A seguir estão as bibliotecas essenciais importadas:

Instalação das Dependências

!pip install optuna optuna-integration[tfkeras]
Optuna: Utilizado para otimização de hiperparâmetros no modelo.

Bibliotecas Importadas

import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
Bibliotecas auxiliares para manipulação de arquivos, data/hora, arrays e visualizações gráficas.

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
Funções para divisão dos dados, pré-processamento, embaralhamento e avaliação do modelo.

import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.saving import register_keras_serializable
Bibliotecas e funções do Keras/TensorFlow para construção do modelo de rede neural e controle do treinamento.

import optuna
from optuna.integration import TFKerasPruningCallback
Optuna: Usado para a otimização de hiperparâmetros durante o treinamento do modelo.
