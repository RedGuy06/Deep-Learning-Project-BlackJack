from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file


app = Flask(__name__)
@app.get("/")
def home():
    return send_file(Path.cwd() / "index.html")


##Photo Path
#path=Path(r"C:\Users\hartmawa\Music\Deep Learning\src\Project\Un_Cropped_Multiple\10.jpg")

#with Image.open(path) as photo:
   # image=np.array(ImageOps.exif_transpose(photo).convert("RGB")) # Convert to numbpy array
    
@app.post("/predict")
def predict():
    photo=request.files["image"]
    
    with Image.open(photo.stream) as photo:
        image=np.array(ImageOps.exif_transpose(photo).convert("RGB")) # Convert to numbpy array


    #Attempt to seperte cards from table, cards should be lighter than table

    cutoff=180 #Mess around to find best image cutoff
    gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #Convert to brightness value
    gray=cv2.GaussianBlur(gray,(1,1),0)#Smoth out image a bit (Probably will need to mess with this value as we go and see what works better)
    _, mask=cv2.threshold(gray,cutoff,255,cv2.THRESH_BINARY) #Create binary cutoff, turn bright pixels white and rest black
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Trance outlines and store them as contour so that we can pull them for later


    #Now that we have the contours we can plot them over the image to check that they are right

    preview=image.copy()#Copy origional image so we can still use it later if we want
    cv2.drawContours(preview,contours,-1,(0,255,0),10) # Draw all contours in green

    #Plot image with contours
    #plt.figure()
    #plt.imshow(preview)


    # Actualy crop the card from the image (Expand to multiple this time)


    card_images=[] #We will put each card into this array
    card_owners=[] #Use this to keep track of dealer vs player card

    split_y=image.shape[0]/2 #Split image in half vertically

    width, height = 224, 320 #Roughly matches actual card aspect ratio
    target = np.float32([[0, 0], [width - 1, 0],[width - 1, height - 1], [0, height - 1]])

    min_area=.01*image.shape[0]*image.shape[1] #Find all of the small contours so that we can ignore them


    

    for card_contour in contours:
        if cv2.contourArea(card_contour)<min_area:
            continue #only go on if contour is big enough to actaully be a card
    #Fit rotated rectangle around the cards
        rect = cv2.minAreaRect(card_contour)
        points = cv2.boxPoints(rect)
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        points = points[np.argsort(angles)]

        # Start on a short edge so the output card is portrait, not sideways.
        side_lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
        if side_lengths[0] + side_lengths[2] > side_lengths[1] + side_lengths[3]:
            points = np.roll(points, -1, axis=0)


        #Map those corners to an upright rectangle and crop from the color photo.


        matrix = cv2.getPerspectiveTransform(points, target)
        crop = cv2.warpPerspective(image, matrix, (width, height))
        card_images.append(Image.fromarray(crop))
        card_owners.append("Dealer" if center[1]<split_y else "Player")

    number_found=len(card_images)
    #print("Cards Found:",number_found)
    #plot_cards
    #for i in range(number_found):
    #   plt.imshow(card_images[i])
    # plt.axis("off")
    
    
    #Now we can just load in the resnet model and run it!!!
    import torch
    from torch import nn
    from torchvision import models, transforms

    #model_path = Path.home() / "models" / "resnet18_realcard_combinedface.pth"
    #model_path = Path.home() / "models" / "resnet18_grayscale_combinedface.pth"
    #model_path = Path.home() / "models" / "resnet18_realcard_androom_combinedface.pth"
    model_path = Path.home() / "models" /"resnet18_two_decks_augmented.pth"

    values=torch.load(model_path,map_location="cpu",weights_only=True)

    #Built the resnest18 with my trained wieghts
    ranks = values["ranks"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(ranks))
    model.load_state_dict(values["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    #Transfroms to match how we trained
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), #Make grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])




    model.eval() #Set model to evaliu
    player_hand=[]
    dealer_hand=[]
    player_value=0 #Initiate player value
    dealer_value=0

    with torch.no_grad():
        for card_image, owner in zip(card_images, card_owners):
            inputs = transform(card_image).unsqueeze(0).to(device)
            prediction = model(inputs).argmax(dim=1).item()
            rank=ranks[prediction]
            
            #Add to either the player or the dealer total depending on classification made earlier
            if owner=="Dealer":
                dealer_hand.append(rank)
            else:
                player_hand.append(rank)



        # plt.figure()
        # plt.imshow(card_image)
        # plt.title(f"Predicted value: {ranks[prediction]}")
        # plt.axis("off")

    def hand_total(hand): #Hand counting method for each case
        total=sum(1 if rank =="A" else int(rank) for rank in hand)#sum all cards in player hand. Assume ace is 1 for now

        if "A" in hand and total+10<=21:
            total+=10 #Make ace 11 if we are able to without busting
        return total

    #Add up each hand
    player_value=hand_total(player_hand)
    dealer_value=hand_total(dealer_hand)

    import pandas as pd
 
    PlayerHand = player_hand
    DealerCard = dealer_hand[0] #Should only be 1 card, but take zero just to make sure
 
    PlayerValue = hand_total(PlayerHand)

    minimum_total = sum(
        1 if card == "A" else int(card)
        for card in PlayerHand
    )

    if PlayerValue > minimum_total:
        sheetName = "Total Dependent - Soft"
    else:
        sheetName = "Total Dependent - Hard"

    if len(PlayerHand) == 2 and PlayerHand[0] == PlayerHand[1]:
        sheetName = "Total Dependent - Split"
        
    print(f"PlayerValue: {PlayerValue}")
    
    print(f'sheetName: {sheetName}')
    
    df = pd.read_excel("DeepLearningLookupTables.xlsx", sheet_name=sheetName)
    
    print(df.head())
    
    # Ensure the key column is the first column
    key_col = "PlayerCard"
    value_col = int(DealerCard) if DealerCard != "A" else "A"
    
    # Create the lookup dictionary
    lookup_dict = df.set_index(key_col)[value_col].to_dict()
    
    # Example usage
    action = lookup_dict.get(PlayerValue)
    
    dictionary = {'H': 'Hit',
                'S': 'Stand',
                'Dh': 'Double if allowed, otherwise Hit',
                'Ds': 'Double if allowed, otherwise stand',
                'P': 'Split',
                'Ph': 'Split if double after split is allowed, otherwise Hit',
                'Pd': 'Split if double after split is allowed, otherwise double and hit',
                'Rh': 'Surrender if allowed, otherwise Hit',
                'Rs': 'Surrender if allowed, otherwise Stand'}
    
   # if action in dictionary:
    #    print(f'Action: {dictionary[action]}')
    #else:
     #   print('Action not found')
    
    recommended_action = dictionary.get(action, "Action not found")

    if player_value > 21:
        recommended_action = "Bust"

    return jsonify(
        player_total=player_value,
        dealer_total=dealer_value,
        action=recommended_action
    )
    

    
app.run(port=5000, use_reloader=False)
