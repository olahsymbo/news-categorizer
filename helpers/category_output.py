import numpy as np

def category_output(prediction):
    if np.argmax(prediction) == 0:
        cate_predict = {"status": "success", "category": "entertainment", "category id": str(np.argmax(prediction)), "prediction rate": str(np.max(prediction)*100)}
    elif np.argmax(prediction) == 1:
        cate_predict = {"status": "success", "category": "sport", "category id": str(np.argmax(prediction)), "prediction rate": str(np.max(prediction)*100)}
    elif np.argmax(prediction) == 2:
        cate_predict = {"status": "success", "category": "politics", "category id": str(np.argmax(prediction)), "prediction rate": str(np.max(prediction)*100)}
    elif np.argmax(prediction) == 3:
        cate_predict = {"status": "success", "category": "business", "category id": str(np.argmax(prediction)), "prediction rate": str(np.max(prediction)*100)}
    elif np.argmax(prediction) == 4:
        cate_predict = {"status": "success", "category": "tech", "category id": str(np.argmax(prediction)), "prediction rate": str(np.max(prediction)*100)}
    return cate_predict