#prediction
import joblib


def predict_price(size,total_sqft,bath,balcony,location):
  model = joblib.load("model.joblib")
  
  location_encoder = joblib.load("location_encoder.joblib")
  available_locations = list(location_encoder.get_feature_names_out())
 
  print(available_locations)
  if "location_"+location not in available_locations:
    location = "other"
  encoded_location = location_encoder.transform([[location]])
  print(encoded_location)
  encoded_location = pd.DataFrame(encoded_location,columns=location_encoder.get_feature_names_out())
  input_data = pd.DataFrame({"size":[size],"total_sqft":[total_sqft],"bath":[bath],"balcony":[balcony]})
  input_data = pd.concat([input_data,encoded_location],axis=1)
  return  model.predict(input_data)

def get_city():
  location_encoder =joblib.load