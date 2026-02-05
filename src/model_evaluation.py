from config import Config

# Import Machine Learning Libraries
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split

def model_evaluation(models, x_train, x_test, y_train, y_test):
    for model_name, model in models.items():
        model.fit(x_train, y_train)  # Seen Data
        y_pred = model.predict(x_test)   # Unseen Data
        print(f"Model: {model_name}")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("-" * 50)

    return classification_report,confusion_matrix
    
