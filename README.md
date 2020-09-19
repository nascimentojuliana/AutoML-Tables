# AutoML-Tables

A **LGPD (Lei Geral de Proteção de Dados)** acaba de ser aprovada e as empresas precisam se adequar às novas normas em relação ao tratamento dos dados pessoais que estão sob sua responsabilidade. Mas, será que as empresas sabem quais são esses dados pessoais? Onde eles estão?

Para ajudar, a **Google** tem uma ferramenta bem simples e interessante, chamada **AutoML Tables**.

Vamos lá. Antes de tudo, você deve conhecer os seus dados e saber exatamente o que precisa obter, ou seja, a resposta que você quer que o seu modelo te dê. No nosso caso, queremos saber se um dado é pessoal ou não.

Para isso, precisamos de um _dataset_ com informações já classificadas. 

Nossa estratégia será utilizar os **metadados** das tabelas para classificar a informação como pessoal ou não pela **LGPD**. Tenho um _dataset_ simples de **metadados**, com nomes de colunas de tabelas, a descrição dos dados contidos naquela coluna e o tipo de dado (_string_, _numbe_r, etc). Você pode acrescentar mais coisas se achar necessário, como o nome da tabela de onde vem aquela coluna.

O objetivo então é dizer se aquela coluna pode conter informações pessoais (por exemplo, uma coluna que tem como nome telefone_cliente claramente contém dados pessoais).

O primeiro passo para treinar o modelo no **AutoML** é definir o seu alvo (target): nesse caso, colocamos Pessoal e Não_pessoal em uma nova coluna no _dataset_, chamada de *Tag*. Você mudar muda o nome *Tag* para outro de sua preferência, apenas lembre de alterar o código antes.

Segue um exemplo de parte do _dataset_ classificado:

![](https://cdn-images-1.medium.com/max/1200/1*TKuoxpG8diDtH6ZlkUBsaw.png)

Beleza, primeiro passo concluído. Temos um arquivo _csv_ (cada linha com o nome de uma coluna das tabelas selecionadas), cada coluna com sua classificação em Pessoal ou Não_pessoal. Quanto mais dados para treinar o modelo, melhor pode ser o desempenho do mesmo. 

O próximo passo é decidir como o modelo vai usar seus dados. Por padrão, o **AutoML** separa os dados de entrada em: 80% para treinamento, 10% para teste e 10% para validação. Você pode deixar o modelo definir isso para você ou pode definir previamente. Após separar seus dados da maneira como preferir, coloque uma coluna a mais (chamei de ML_USE) e diga se quer que aquele dado seja usado para TRAIN, TEST ou VALIDATION. Escreva exatamente dessa forma.

Com seu _dataset_ pronto, é hora de começar.

Abra seu editor favorito e use o código disponívle em *src*. Pode ir fazendo o passo a passo em um _Jupyter Notebook_ também. Aqui, temos uma explicação bem legal de como fazer cada etapa (https://medium.com/@juliacabe/lgpd-64b6eaeed826?sk=65c69761eca86eec1b72e197f263d71d).

Como entrada no arquivo .sh você vai ter:

--project_id "" , onde você deve definir o projeto que está usando no GCP
--compute_region '' , aqui deve ser us-central1
--input '' , aqui você deve inserir o arquivo que será usado para treinar o modelo, em formato csv
--bucket '' , nesse código, vamos salvar o arquivo de treinamento no storage, por isso defina aqui o bucket que será usado
--model '' , só dizer o nome que você quer que o seu modelo tenha
--output_path '' como o arquivo será salvo no bucket, ex: input/dataset.csv
