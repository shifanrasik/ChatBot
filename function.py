def contains_promotion(sentence):
    promotion_words = ["discount", "promo", "offer", "deal"]
    
    for word in promotion_words:
        if word in sentence.lower():
            return True
    
    return False