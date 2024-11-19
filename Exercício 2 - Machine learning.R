###############################
#Pacotes
###############################
install.packages("tm") #tratar dados de texto
install.packages("SnowballC")
install.packages("wordcloud")
install.packages("e1071")
install.packages("caret") #matrix de confusão

library(tm)
library(SnowballC)
library(wordcloud)
library(e1071)
library(caret)

tweets <- Exercício_2_Tweets_Mg[, c("Text", "Classificacao")]

################################################################
# Parte 1 - criar um corpus (conjunto de documentos de texto)
################################################################

tweets_corpus <- VCorpus(VectorSource(tweets$Text))

############################################
# Parte 2 - Limpar os documentos de texto
############################################

# 1 - transformar tudo em letra minúscula
tweets_limpo <- tm_map(tweets_corpus, content_transformer(tolower))

# Comparando
as.character(tweets_corpus[[1]])
as.character(tweets_limpo[[1]])

# 2 - Remover números
tweets_limpo <- tm_map(tweets_limpo, removeNumbers)

# 3 - Remover stopwords
tweets_limpo <- tm_map(tweets_limpo, removeWords, stopwords(kind = "pt-br")) #(kind = "pt-br")) dentro de stopword

# 4 - Remover pontuações
tweets_limpo <- tm_map(tweets_limpo, removePunctuation)

# 5 - Remover espaços em excesso
tweets_limpo <- tm_map(tweets_limpo, stripWhitespace)

# 6 - Manter os radicais
tweets_limpo <- tm_map(tweets_limpo, stemDocument)

##########################
# Parte 3 - Tokenização
##########################

# Criando a matriz de termos dos documentos(DTM)
tweets_dtm <- DocumentTermMatrix(tweets_limpo)

#Separando os dados em treino e testeS

sorteio <- sample(1:8199, 6560)

tweets_dtm_treino <- tweets_dtm[sorteio,]
tweets_dtm_teste <- tweets_dtm[-sorteio,]

tweets_dtm_rotulos_treino <- tweets[sorteio,]$Classificacao
tweets_dtm_rotulos_teste <- tweets[-sorteio,]$Classificacao

# Visualização em nuvem de palavras
wordcloud(tweets_limpo, min.freq = 50, random.order = FALSE)

spam <- subset(tweets, type = "spam")
wordcloud(spam$Text, min.freq = 40, random.order = FALSE, scale = c(3, 0.5))

ham <- subset(tweets, type = "ham")
wordcloud(ham$Text, min.freq = 40, random.order = FALSE, scale = c(3, 0.5))

# Deixando apenas palavras com frequencia >= 5
tweets_palavras_freq_treino <- findFreqTerms(tweets_dtm_treino, 5)
tweets_palavras_freq_teste <- findFreqTerms(tweets_dtm_teste, 5)

tweets_dtm_freq_treino <- tweets_dtm_treino[, tweets_palavras_freq_treino]
tweets_dtm_freq_teste <- tweets_dtm_teste[, tweets_palavras_freq_teste]

# Finalmente, contando as palavras em cada mensagem
converter_contagens <- function(x) {
  x <- ifelse(x > 0, "Sim", "Não")
}

tweets_treino <- apply(tweets_dtm_freq_treino, MARGIN = 2, converter_contagens)
tweets_teste <- apply(tweets_dtm_freq_teste, MARGIN = 2, converter_contagens)

###########################
# Parte 4 - Classificador
###########################

#treino
tweets_classificador <- naiveBayes(tweets_treino, tweets_dtm_rotulos_treino)

#teste
tweets_teste_pred <- predict(tweets_classificador, tweets_teste)























