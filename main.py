from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_build import model_building
from src.model_evaluation import model_evaluation

def main():
    df = data_ingestion()
    x_train, x_test, y_train, y_test = data_preprocessing(df)
    models = model_building(df)
    classification_report, confusion_matrix = model_evaluation(models, x_train, x_test, y_train, y_test)
    print(classification_report)
   
if __name__ == "__main__":
    main()
