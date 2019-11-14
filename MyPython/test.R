library(klaR)
model <- rda(Species~., data = train.transformed)

predictions <- model %>% predict(test.transformed)

mean(predictions$class == test.transformed$Species)