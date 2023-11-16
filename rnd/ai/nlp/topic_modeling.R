#'
#'
#' https://github.com/sibanjan/text_mining/blob/master/tm_topic_model.R
#' https://dzone.com/articles/text-mining-using-r-and-h2o-let-machine-learn-lang
#'
suppressPackageStartupMessages(suppressWarnings({
  library(argparse)
  parser <- ArgumentParser()
  parser$add_argument("--job_home", default = "~/dev/xinh3ng/DSResearch/ML/gist")
  args <- parser$parse_args()

  setwd(args$job_home)  # set working dir
  Sys.setenv(TZ = "America/Los_Angeles")  # remove "unknown timezone warning"
}))
suppressPackageStartupMessages(suppressWarnings({
  library(dplyr)
  library(tm)
  library(textir)
  library(SnowballC)
  library(topicmodels)

  library(futile.logger)  # Set up the logger
  flog.layout(layout.format("~t ~l ~n.~f: ~m"))
  flog.threshold(futile.logger::INFO)  # DEBUG, INFO
}))

# Load files into corpus
filenames <- list.files(paste0(getwd(), "/data"), pattern = "*.txt", full.names = TRUE)
data <- lapply(filenames, readLines)
corpus <- Corpus(VectorSource(data))
corpus <- tm_map(corpus, tolower)  # make each letter lowercase
corpus <- tm_map(corpus, removePunctuation)  # remove punctuation
corpus <- tm_map(corpus, removeNumbers)  #remove numbers

# remove generic and custom stopwords
stop_words <- c(stopwords("english"), "best")
corpus <- tm_map(corpus, removeWords, stop_words)
corpus <- tm_map(corpus, stemDocument)

myDtm = DocumentTermMatrix(corpus, control = list(minWordLength = 3));
mydata.dtm2 <- removeSparseTerms(mydata.dtm, sparse=0.98)

k = 2;
SEED = 1234;
my_TM =
  list(VEM = LDA(myDtm, k = k, control = list(seed = SEED)),
       VEM_fixed = LDA(myDtm, k = k,
                       control = list(estimate.alpha = FALSE, seed = SEED)),
       Gibbs = LDA(myDtm, k = k, method = "Gibbs",
                   https://www.r-bloggers.com/text-mining/
                     https://www.slideshare.net/MinhaHwang/introduction-to-text-mining-32058520

                   findFreqTerms(myDtm, lowfreq=50);
                   #find the probability a word is associated
                   findAssocs(myDtm, 'find_a_word', 0.5);

# Stop everything
flog.info("ALL DONE\n")
