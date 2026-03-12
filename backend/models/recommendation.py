def get_recommendation(risk):

    if risk == "Low":
        return "Your lifestyle looks healthy. Maintain current habits."

    elif risk == "Medium":
        return "Increase hydration and reduce screen time."

    else:
        return "Improve sleep, reduce stress, increase hydration and exercise."