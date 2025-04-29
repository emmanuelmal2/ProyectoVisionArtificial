from tensorflow.keras.preprocessing.image import ImageDataGenerator

def cargar_datos_generador(base_path, img_size=(128, 128), batch_size=32):
    """Carga datos en lotes usando generadores, dividiendo en entrenamiento y validación."""
    # Crear generador de datos con normalización y división
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Generador para el conjunto de entrenamiento
    train_generator = datagen.flow_from_directory(
        base_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',  # Las etiquetas se devuelven como índices
        subset='training'  # Parte del conjunto de entrenamiento
    )

    # Generador para el conjunto de validación
    validation_generator = datagen.flow_from_directory(
        base_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode='sparse',
        subset='validation',
        shuffle=False  # Mantiene el orden
    )


    return train_generator, validation_generator


categoria_mapeo = {
    "Apple": [
        "Apple 6", "Apple Braeburn 1", "Apple Crimson Snow 1", "Apple Golden 1",
        "Apple Golden 2", "Apple Golden 3", "Apple Granny Smith 1", "Apple hit 1",
        "Apple Pink Lady 1", "Apple Red 1", "Apple Red 2", "Apple Red 3",
        "Apple Red Delicious 1", "Apple Red Yellow 1", "Apple Red Yellow 2"
    ],
    "Banana": [
        "Banana 1", "Banana Lady Finger 1", "Banana Red 1"
    ],
    "Apricot": ["Apricot 1"],
    "Avocado": ["Avocado 1", "Avocado ripe 1"],
    "Cantaloupe": ["Cantaloupe 1", "Cantaloupe 2"],
    "Blueberry": ["Blueberry 1"],
    "Cabbage": ["Cabbage white 1"],
    "Cactus": ["Cactus fruit 1"],
    "Carrot": ["Carrot 1"],
    "Cauliflower": ["Cauliflower 1"],
    "Cherry": [
        "Cherry 1", "Cherry 2", "Cherry Rainier 1", "Cherry Wax Black 1",
        "Cherry Wax Red 1", "Cherry Wax Yellow 1"
    ],
    "Chestnut": ["Chestnut 1"],
    "Clementine": ["Clementine 1"],
    "Coconut": ["Cocos 1"],
    "Corn": ["Corn 1", "Corn Husk 1"],
    "Cucumber": [
        "Cucumber 1", "Cucumber 2", "Cucumber 3", "Cucumber Ripe 1",
        "Cucumber Ripe 2"
    ],
    "Dates": ["Dates 1"],
    "Eggplant": ["Eggplant 1", "Eggplant long 1"],
    "Fig": ["Fig 1"],
    "Ginger": ["Ginger Root 1"],
    "Granadilla": ["Granadilla 1"],
    "Grape": [
        "Grape Blue 1", "Grape Pink 1", "Grape White 1", "Grape White 2",
        "Grape White 3", "Grape White 4"
    ],
    "Grapefruit": ["Grapefruit Pink 1", "Grapefruit White 1"],
    "Guava": ["Guava 1"],
    "Hazelnut": ["Hazelnut 1"],
    "Huckleberry": ["Huckleberry 1"],
    "Kaki": ["Kaki 1"],
    "Kiwi": ["Kiwi 1"],
    "Kohlrabi": ["Kohlrabi 1"],
    "Kumquats": ["Kumquats 1"],
    "Lemon": ["Lemon 1", "Lemon Meyer 1"],
    "Lime": ["Limes 1"],
    "Lychee": ["Lychee 1"],
    "Mandarine": ["Mandarine 1"],
    "Mango": ["Mango 1", "Mango Red 1"],
    "Mangostan": ["Mangostan 1"],
    "Maracuja": ["Maracuja 1"],
    "Melon": ["Melon Piel de Sapo 1"],
    "Mulberry": ["Mulberry 1"],
    "Nectarine": ["Nectarine 1", "Nectarine Flat 1"],
    "Nut": ["Nut Forest 1", "Nut Pecan 1"],
    "Onion": ["Onion Red 1", "Onion Red Peeled 1", "Onion White 1"],
    "Orange": ["Orange 1"],
    "Papaya": ["Papaya 1"],
    "Passion Fruit": ["Passion Fruit 1"],
    "Peach": ["Peach 1", "Peach 2", "Peach Flat 1"],
    "Pear": [
        "Pear 1", "Pear 2", "Pear 3", "Pear Abate 1", "Pear Forelle 1",
        "Pear Kaiser 1", "Pear Monster 1", "Pear Red 1", "Pear Stone 1",
        "Pear Williams 1"
    ],
    "Pepper": [
        "Pepper Green 1", "Pepper Orange 1", "Pepper Red 1", "Pepper Yellow 1"
    ],
    "Physalis": ["Physalis 1", "Physalis with Husk 1"],
    "Pineapple": ["Pineapple 1", "Pineapple Mini 1"],
    "Pitahaya": ["Pitahaya Red 1"],
    "Plum": ["Plum 1", "Plum 2", "Plum 3"],
    "Pomegranate": ["Pomegranate 1"],
    "Pomelo": ["Pomelo Sweetie 1"],
    "Potato": [
        "Potato Red 1", "Potato Red Washed 1", "Potato Sweet 1", "Potato White 1"
    ],
    "Quince": ["Quince 1"],
    "Rambutan": ["Rambutan 1"],
    "Raspberry": ["Raspberry 1"],
    "Redcurrant": ["Redcurrant 1"],
    "Salak": ["Salak 1"],
    "Strawberry": ["Strawberry 1", "Strawberry Wedge 1"],
    "Tamarillo": ["Tamarillo 1"],
    "Tangelo": ["Tangelo 1"],
    "Tomato": [
        "Tomato 1", "Tomato 2", "Tomato 3", "Tomato 4", "Tomato Cherry Red 1",
        "Tomato Heart 1", "Tomato Maroon 1", "Tomato not Ripened 1",
        "Tomato Yellow 1"
    ],
    "Walnut": ["Walnut 1"],
    "Watermelon": ["Watermelon 1"],
    "Zucchini": ["Zucchini 1", "Zucchini dark 1"]
}
