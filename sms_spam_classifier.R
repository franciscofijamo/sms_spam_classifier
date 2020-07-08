library(tm) # text mining
library(SnowballC) 
library(wordcloud) # viz wordcloud
library(e1071) # naive B package
library(gmodels) #confusion matriz
library(corpus)

# Carregamento dos dados
dados <- read.csv("sms_spam.csv", stringsAsFactors = FALSE)

# Conhencendo os dados
head(dados)
dim(dados)
str(dados)

# Convertendo variavel type para factor
dados$type <-factor(dados$type)
str(dados$type)
# tabela de contigencia
table(dados$type)

# Contruindo o Corpus do da var teste
dados_corpus <- VCorpus(VectorSource(dados$text))

# visualizar 
print(dados_corpus)

# ispencao do resumo dos dados
inspect(dados_corpus[1:2])

# Ajustar estrutura
as.character(dados_corpus[[1]])

# aplicando lapply
lapply(dados_corpus[1:2], as.character)



# Limpeza dos dados com tm_map()
?tm_map
# Converter todo texto para minusculas
dados_corpus_clean <- tm_map(dados_corpus, content_transformer(tolower))


# pre vislualizacoa dos dados antes e depois

as.character(dados_corpus[[1]])
as.character(dados_corpus_clean[[1]])

# continuando na limpeza dos dados
# Removendo numeros
dados_corpus_clean <- tm_map(dados_corpus_clean, removeNumbers)
# Removendo stopWords
dados_corpus_clean <- tm_map(dados_corpus_clean, removeWords,  stopwords())
# Removendo pontuacoes
dados_corpus_clean <- tm_map(dados_corpus_clean, removePunctuation)

# Criar uma funcao para subistituir pontuacao, e nao remover.
# se voce apenas remover  as pontuacoes, a estrutura do texto pode mudar significadamente


removePunctuation("hello...Mozambique")
replacePontuation <- function(x){gsub("[[:punct:]]+", " ", x)}
replacePontuation("hello...Mozambique")


#Word Steming

# tratar palavras "sinonimas" ou parecidas
?wordStem
# exemplo :
#wordStem(c("learn", "learned","learning","learns"))

# Aplicando Stem
dados_corpus_clean <- tm_map(dados_corpus_clean, stemDocument)

# Eliminando espcacos em branco desnecessarios, ou duplos.
dados_corpus_clean <- tm_map(dados_corpus_clean, stripWhitespace)

# Visualizando o antese depois
lapply(dados_corpus[1:3], as.character)
lapply(dados_corpus_clean[1:3], as.character)


# Contruindo 3 versoes de matrizes
?DocumentTermMatrix
# v1
dados_doc_ter_mx <- DocumentTermMatrix(dados_corpus_clean)


# v2
dados_doc_ter_mx_v2 <- DocumentTermMatrix(dados_corpus, control = list(tolower = TRUE,
                                                                       removeNumbers = TRUE,
                                                                       stopwords = TRUE,
                                                                       removePunctuation = TRUE,
                                                                       steming = TRUE
))
# v3
dados_doc_ter_mx_v3 <- DocumentTermMatrix(dados_corpus, control = list(tolower = TRUE,
                                                                       removeNumbers = TRUE,
                                                                       stopwords = function(x){removeWords(x,stopwords())},
                                                                       removePunctuation = TRUE,
                                                                       steming = TRUE
))


# Comparar Resultados
dados_doc_ter_mx
dados_doc_ter_mx_v2
dados_doc_ter_mx_v3

# O primeiro gerou o menor numero de termos, o que significa que seu pre-preocessamento
# apresentou melhor exito, sendo que os dois ultimos apresentaram mais termos, que pode
# significar que o processo nao soube lhe dar melhor com alguns stopwords, por isso retornou
# mais termos que a primeira versao.


#Machine Learning


# Criando dados de treino  e teste
dados_X_treino <- dados_doc_ter_mx[1:4169, ]
dados_X_teste <- dados_doc_ter_mx[4170:5559, ]


# Criando variavel target, treino  e teste
dados_Y_treino <- dados[1:4169, ]$type
dados_Y_teste <- dados[4170:5559, ]$type

# verificando porporcoes de dados de spam
prop.table(table(dados_Y_treino))
prop.table(table(dados_Y_teste))


# WorldCloud viz

wordcloud(dados_corpus_clean, min.freq = 10, random.order = FALSE)


# Frequencia de Dados
# Coletar as palavras que aparecem mais frequentemente, ou seja reduzir a matriz esparsa

sns_dados_train <- removeSparseTerms(dados_X_treino, 0.999)
sns_dados_train


# indicador de features para palavras mais frequentes
sns_palavras_freq <- findFreqTerms(dados_X_treino, 5)
str(sns_palavras_freq)

#sremover palavras que especificas que sao residuos
sns_palavras_freq <- sns_palavras_freq[-c(1:2)]


# Criando subsets com palavras mais frequentes
sns_palavras_freq_treino <- dados_X_treino[, sns_palavras_freq]
sns_palavras_freq_teste <- dados_X_teste[, sns_palavras_freq]

# convertendo a contagem para factor
converter_contagem <- function(x){ 
  x <- ifelse(x>0, "SSPAM", "No-SPAM")
}

# converter converter_contagem para colunas de dados de treino e de teste
sns_train <- apply(sns_palavras_freq_treino, MARGIN = 2, converter_contagem)
sns_teste <- apply(sns_palavras_freq_teste, MARGIN = 2, converter_contagem)



# Treiando o modelo

classificados_naiveb <- naiveBayes(sns_train, dados_Y_treino)


# Avaliando o modelo 
sns_predict <- predict(classificados_naiveb, sns_teste)


# Contruindo uma confusion Matriz para avaliar  o modelo 
CrossTable(sns_predict,
           dados_Y_teste,
           prop.chisq = FALSE,
           prop.t = FALSE,
           prop.r = FALSE,
           dnn = c("Previsao", "Resultado"))

# Melhorando a performace do modelo, aplicando uavizacao com laplace = 1
# quando uma palavra nao for encontrada, aplicara probabilidade 1, invez de 0
classificados_naiveb_v2 <- naiveBayes(sns_train, dados_Y_treino, laplace = 1)


# Avaliando o modelo v2
sns_predict_2 <- predict(classificados_naiveb_v2, sns_teste)


# Contruindo uma confusion Matriz para avaliar a versao 2 do modelo 
CrossTable(sns_predict_2,
           dados_Y_teste,
           prop.chisq = FALSE,
           prop.t = FALSE,
           prop.r = FALSE,
           dnn = c("Previsao", "Resultado"))


