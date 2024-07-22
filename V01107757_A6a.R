install.packages("imputeTS")

# Load necessary libraries
library(tidyverse)
library(lubridate)
library(imputeTS)
library(forecast)
library(ggplot2)

setwd("C:\\A6a")
# Load the dataset
amz_data <- read.csv("C:\\A6a\\EBAY Historical Data.csv")

# Convert the Date column to Date type using parse_date_time function
amz_data$Date <- parse_date_time(amz_data$Date, orders = c("mdy", "dmy", "ymd"))

# Check for missing dates and remove rows with missing dates
amz_data <- amz_data %>% drop_na(Date)


# Convert Vol. and Change % from string to numeric
amz_data$Vol. <- as.numeric(gsub("M", "", amz_data$Vol.)) * 1e6
amz_data$Change. <- as.numeric(gsub("%", "", amz_data$Change.))



# Check for missing values in other columns
sum(is.na(amz_data))


# Interpolate missing values if there are any
amz_data <- na_interpolation(amz_data)


# Check for outliers using boxplot
boxplot(amz_data$Price, main="Boxplot for Price", ylab="Price")


# Plot the data
ggplot(amz_data, aes(x = Date, y = Price)) +
  geom_line() +
  labs(title = "EBAY Stock Price Over Time", x = "Date", y = "Price")



# Split the data into training and testing sets
split_date <- as.Date("2023-12-31")
train_data <- amz_data %>% filter(Date <= split_date)
test_data <- amz_data %>% filter(Date > split_date)


# Convert the data to monthly
amz_data_monthly <- amz_data %>%
  group_by(month = floor_date(Date, "month")) %>%
  summarise(Price = mean(Price))


# Create time series object
amz_ts <- ts(amz_data_monthly$Price, start = c(2020, 1), frequency = 12)


# Decompose the time series using additive model
decomp_additive <- decompose(amz_ts, type = "additive")


# Decompose the time series using multiplicative model
decomp_multiplicative <- decompose(amz_ts, type = "multiplicative")



# Plot the decomposed components for additive model
autoplot(decomp_additive) +
  ggtitle("Additive Decomposition of EBAY Stock Price") +
  theme_minimal()


# Plot the decomposed components for multiplicative model
autoplot(decomp_multiplicative) +
  ggtitle("Multiplicative Decomposition of EBAY Stock Price") +
  theme_minimal()


# Print a message to indicate completion
print("Data cleaning, interpolation, plotting, and decomposition are complete.")




## UNIVARIATE ANALYSIS

# Create time series objects
amz_ts_daily <- ts(amz_data$Price, start = c(2020, 1), frequency = 365.25)
amz_ts_monthly <- ts(amz_data_monthly$Price, start = c(2020, 1), frequency = 12)


# 1. Holt-Winters model and forecast for the next year
hw_model <- HoltWinters(amz_ts_monthly)
hw_forecast <- forecast(hw_model, h = 12)
autoplot(hw_forecast) +
  ggtitle("Holt-Winters Forecast for Amazon Stock Price") +
  theme_minimal()



# 2. Fit ARIMA model to the daily data
arima_model_daily <- auto.arima(amz_ts_daily)
summary(arima_model_daily)


# Diagnostic check for ARIMA model
checkresiduals(arima_model_daily)


# Fit SARIMA model to the daily data
sarima_model_daily <- auto.arima(amz_ts_daily, seasonal = TRUE)
summary(sarima_model_daily)


# Compare ARIMA and SARIMA models
arima_aic <- AIC(arima_model_daily)
sarima_aic <- AIC(sarima_model_daily)
print(paste("ARIMA AIC:", arima_aic))
print(paste("SARIMA AIC:", sarima_aic))


# Select the best model based on AIC
if (arima_aic < sarima_aic) {
  best_model_daily <- arima_model_daily
} else {
  best_model_daily <- sarima_model_daily
}


# Ensure the best model is a valid forecast model
if (!inherits(best_model_daily, "Arima")) {
  stop("The selected best model is not a valid ARIMA model")
}

# Forecast for the next 90 days
daily_forecast <- forecast(best_model_daily, h = 90)



# Check if forecast object is created correctly
if (!inherits(daily_forecast, "forecast")) {
  stop("Forecast object was not created correctly")
}


# Plot the forecast
autoplot(daily_forecast) +
  ggtitle("Daily Forecast for EBAY Stock Price") +
  theme_minimal()



# 3. Fit ARIMA model to the monthly series
arima_model_monthly <- auto.arima(amz_ts_monthly)
summary(arima_model_monthly)


# Forecast the monthly series
monthly_forecast <- forecast(arima_model_monthly, h = 12)
autoplot(monthly_forecast) +
  ggtitle("Monthly ARIMA Forecast for EBAY Stock Price") +
  theme_minimal()



## MULTIVARIATE

install.packages("keras")
install.packages("randomForest")
library(keras)
library(randomForest)
library(rpart)
library(caret)


# Feature engineering: Create lagged variables and moving averages as features
amz_data <- amz_data %>%
  mutate(Price_lag1 = lag(Price, 1),
         Price_lag2 = lag(Price, 2),
         Price_lag3 = lag(Price, 3),
         Vol_lag1 = lag(Vol., 1),
         Vol_lag2 = lag(Vol., 2),
         Vol_lag3 = lag(Vol., 3),
         Price_ma7 = rollmean(Price, 7, fill = NA, align = "right"),
         Price_ma30 = rollmean(Price, 30, fill = NA, align = "right"))

# Remove rows with NA values generated by lagging
amz_data <- amz_data %>% drop_na()


# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(amz_data$Price, p = 0.8, list = FALSE)
train_data <- amz_data[train_index, ]
test_data <- amz_data[-train_index, ]


# Convert features to numeric
train_features <- train_data %>% select(-Date, -Price) %>% mutate(across(everything(), as.numeric))
test_features <- test_data %>% select(-Date, -Price) %>% mutate(across(everything(), as.numeric))


# Check if the number of columns match
if (ncol(train_features) != ncol(test_features)) {
  stop("The number of columns in train_features and test_features do not match.")
}


# Normalize the features
train_features <- scale(train_features)
test_features <- scale(test_features, center = attr(train_features, "scaled:center"), scale = attr(train_features, "scaled:scale"))


# Prepare labels
train_labels <- train_data$Price
test_labels <- test_data$Price


# Reshape the data for LSTM input (samples, time steps, features)
train_array <- array(train_features, dim = c(nrow(train_features), 1, ncol(train_features)))
test_array <- array(test_features, dim = c(nrow(test_features), 1, ncol(test_features)))


# Build and train the LSTM model
lstm_model <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(1, ncol(train_features)), return_sequences = TRUE) %>%
  layer_lstm(units = 50) %>%
  layer_dense(units = 1)


history <- lstm_model %>% fit(
  train_array, train_labels,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2
)

# Forecast using the LSTM model
lstm_forecast <- lstm_model %>% predict(test_array)

# Plot the LSTM forecast
plot(test_labels, type = "l", col = "blue", main = "LSTM Forecast vs Actual", xlab = "Time", ylab = "Price")
lines(lstm_forecast, col = "red")
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1)

# Build and train the Random Forest model
rf_model <- randomForest(Price ~ ., data = train_data %>% select(-Date))
rf_forecast <- predict(rf_model, test_data %>% select(-Date, -Price))

# Plot the Random Forest forecast
plot(test_labels, type = "l", col = "blue", main = "Random Forest Forecast vs Actual", xlab = "Time", ylab = "Price")
lines(rf_forecast, col = "red")
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1)

# Build and train the Decision Tree model
dt_model <- rpart(Price ~ ., data = train_data %>% select(-Date))
dt_forecast <- predict(dt_model, test_data %>% select(-Date, -Price))

# Plot the Decision Tree forecast
plot(test_labels, type = "l", col = "blue", main = "Decision Tree Forecast vs Actual", xlab = "Time", ylab = "Price")
lines(dt_forecast, col = "red")
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1)
