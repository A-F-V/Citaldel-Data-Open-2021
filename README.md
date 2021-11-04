# Welcome to Team 8's Europe Citadel Data Open Submission

---
# To Setup
1. `pip install virtualenv`
2. Create Environment for first time `python -m venv env`
3. Enter environment with `source env/bin/activate` on unix or `.\env\Scripts\activate` on windows
4. Install requirements `pip install -r requirements.txt`


# To run
1. Enter environment with `source env/bin/activate` on unix or `.\env\Scripts\activate` on windows
2. Run files normally

---
# File Explanation
- [Raw Data](data/raw) - All the raw data including lexicons and packages
- [Processed Data](data/processed) - The processed dat
- [Headline To Emotion](src/headline_to_emotion.py) - **Takes the packages data set and extracts the emotions from all the data**


---
# TODO
- [x] Also carry the mean for a test with each x
- [ ] Create a random forest model with a linear model to predict change in clicks (or equiv)
- [ ] Create a ANN to do same as above