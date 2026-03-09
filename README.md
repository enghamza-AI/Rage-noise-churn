# rage-noise-churn

**Tagline:** Toxic rage reviews poison churn prediction — watch coefficients flip and error explode

### What this project shows

In mobile games, real gameplay (wins, kills, playtime) should predict player satisfaction perfectly.  
But angry players leave random low reviews (rage noise) that can ruin your model.

This experiment:
- Starts with clean satisfaction scores based on gameplay
- Adds increasing "rage noise" to training scores
- Trains a simple linear model to predict satisfaction
- Watches when the model starts believing stupid things (e.g. "more wins = lower satisfaction")

