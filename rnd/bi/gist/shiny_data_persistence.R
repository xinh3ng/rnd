#'
#' Link: https://shiny.rstudio.com/articles/persistent-data-storage.html
#' 
suppressMessages({
  library(dplyr)
  library(futile.logger)
  library(RMySQL)
  library(rpchutils)
  library(shiny)
  
  flog.threshold(futile.logger::INFO)
  unused <- flog.layout(layout.format("~t ~l ~n.~f: ~m"))
  
  if (Sys.getenv("PROJ_HOME") == "") {
    # Idendify current file location in RStudio environment
    Sys.setenv("PROJ_HOME" = dirname(rstudioapi::getActiveDocumentContext()$path))
  }
  setwd(Sys.getenv("PROJ_HOME"))
  flog.info("Setting the project home: %s", Sys.getenv("PROJ_HOME"))
})

suppressMessages({
  library(argparse)
  parser <- ArgumentParser()
  
  parser$add_argument("--user_host", default = "developers@redshift.punchh.com")
  args <- parser$parse_args()
})

options(mysql = list(
  "host" = "127.0.0.1",
  "port" = 3306,
  "user" = "myuser",
  "password" = "mypassword"
))

dbname <- "myshinydb"
table <- "responses"

#' Define the fields we want to save from the form
fields <- c("name", "used_shiny", "r_years")

#########################################################

#' Save data to a response
#' This is one of the two functions we will change for every storage type
save_data_memory <- function(data) {
  data <- as.data.frame(t(data))
  if (exists("responses")) {
    responses <<- dplyr::bind_rows(responses, data)
  } else {
    responses <<- data
  }
}

# Load all previous responses
# This is one of the two functions we will change for every storage type
load_data_memory <- function() {
  if (exists("responses")) {
    responses
  }
}

#' Save data to a data table
#' 
save_data_mysql <- function(data) {
  # Connect to the database
  conn <- dbConnect(MySQL(), dbname = dbname, host = options()$mysql$host, 
                  port = options()$mysql$port, user = options()$mysql$user, 
                  password = options()$mysql$password)

  query <- sprintf(
    "INSERT INTO %s (%s) VALUES ('%s')",
    table, 
    paste(names(data), collapse = ", "),
    paste(data, collapse = "', '")
  )
  # Submit the update query and disconnect
  dbGetQuery(conn, query)
  dbDisconnect(conn)
}

#'
load_data_mysql <- function() {
  # Connect to the database
  conn <- dbConnect(MySQL(), dbname = dbname, host = options()$mysql$host, 
                  port = options()$mysql$port, user = options()$mysql$user, 
                  password = options()$mysql$password)

  # Construct the fetching query
  query <- sprintf("SELECT * FROM %s", table)
  data <- dbGetQuery(conn, query)
  dbDisconnect(conn)
  flog.info("Successfully got all the data and disonnected from the database")
  data
}

#########################################################

save_data <- save_data_memory
load_data <- load_data_memory
  

# The Shiny app with 3 fields that the user can submit data for
shinyApp(
  ui = fluidPage(
    DT::dataTableOutput("responses", width = 400), tags$hr(),
  
    textInput("name", "Your Name", ""),
    checkboxInput("used_shiny", "Check the box if you have built a Shiny app before", FALSE),
    sliderInput("r_years", "R experience (years)", 0, 25, 2, ticks = FALSE),
    
    actionButton("submit", "Submit")
  ),
  
  server = function(input, output, session) {
    
    # Whenever a field is filled, aggregate all form data
    form_data <- reactive({
      data <- sapply(fields, function(x) input[[x]])
      data
    })
    
    # When the Submit button is clicked, save the form data
    observeEvent(input$submit, {
      save_data(form_data())
    })
    
    # Show the previous responses
    # update with current response when Submit is clicked
    output$responses <- DT::renderDataTable({
      input$submit
      load_data()
    })     
  }
)

flog.info("ALL DONE\n")
