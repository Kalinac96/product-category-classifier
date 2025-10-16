import joblib

model = joblib.load("models/product_classifier.pkl")

print("Interaktivna klasifikacija proizvoda. Unesi naziv proizvoda ili 'exit' za izlaz.\n")

while True:
    title = input("Unesi naziv proizvoda: ")
    if title.lower() == "exit":
        print("Izlaz iz programa. Doviđenja!")
        break
    prediction = model.predict([title])
    print(f"Predviđena kategorija: {prediction[0]}\n")