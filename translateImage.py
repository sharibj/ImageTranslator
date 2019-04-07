# Author - Sharib Jafari
# April 2019

# Code to detect Text and OCR is taken from the URL below
# https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/
# It is an awesome tutorial and my inspiration for this project


# USAGE
# python translateImage.py --east frozen_east_text_detection.pb --image images/example_01.jpg
# python translateImage.py --east frozen_east_text_detection.pb --image images/example_04.jpg --padding 0.05

# import the necessary packages
from imutils.object_detection import non_max_suppression
# Google Translator
from googletrans import Translator
# PIL for writing on image in fonts other than english
from PIL import ImageFont, ImageDraw, Image
# Clustering is needed to find dominant color
from sklearn.cluster import KMeans
from collections import Counter
#from translation import google, ConnectError
import numpy as np
import pytesseract
import argparse
import cv2

# get_dominant_color is taken as is from the URL below
# Except maybe some very little changes
# https://adamspannbauer.github.io/2018/03/02/app-icon-dominant-colors/
def get_dominant_color(image, k=4, image_processing_size = None):
    """
    takes an image as input
    returns the dominant color of the image as a list
    
    dominant color is found by running k means on the 
    pixels & returning the centroid of the largest cluster

    processing time is sped up by working with a smaller image; 
    this resizing can be done with the image_processing_size param 
    which takes a tuple of image dims as input

    >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
    [56.2423442, 34.0834233, 70.1234123]
    """
    #resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, 
                            interpolation = cv2.INTER_AREA)
    
    #reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    #cluster and assign labels to the pixels 
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)

    #count labels to find most popular
    label_counts = Counter(labels)

    #subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return dominant_color


def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320,
	help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(origH, origW) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# decode the predictions, then  apply non-maxima suppression to
# suppress weak, overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

# initialize the list of results
results = []

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective
	# ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	# in order to obtain a better OCR of the text we can potentially
	# apply a bit of padding surrounding the bounding box -- here we
	# are computing the deltas in both the x and y directions
	dX = int((endX - startX) * args["padding"])
	dY = int((endY - startY) * args["padding"])

	# apply padding to each side of the bounding box, respectively
	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(origW, endX + (dX * 2))
	endY = min(origH, endY + (dY * 2))

	# extract the actual padded ROI
	roi = orig[startY:endY, startX:endX]

	# in order to apply Tesseract v4 to OCR text we must supply
	# (1) a language, (2) an OEM flag of 4, indicating that the we
	# wish to use the LSTM neural net model for OCR, and finally
	# (3) an OEM value, in this case, 7 which implies that we are
	# treating the ROI as a single line of text
	config = ("-l eng --oem 1 --psm 7 -c preserve_interword_spaces=1")
	text = pytesseract.image_to_string(roi, config=config)
	# add the bounding box coordinates and OCR'd text to the list
	# of results
	results.append(((startX, startY, endX, endY), text))

# Sharib ............................................................................................

# Translator works better if we translate whole sentences instead of individual words
# "results" is a list of individual words
# Here I'll group these words togather if they belong to the same line (based on their coordinates)
# @TODO : Can this be achieved without as many nested loops?

# This is array of arrays
# Each child array will contain words belonging to the same line
groupedResults = []
# margin of y-axis variation that can be considered as a single line
margin = 5
for ((startX, startY, endX, endY), text) in results:
	addFlag = True
	for resultSet in groupedResults:
		# Y-axis difference
		diffY = abs(startY - resultSet[0][0][1])
		if diffY < margin:
			result = ((startX, startY, endX, endY), text)
			resultSet.append(result)
			addFlag = False
			break
	if(addFlag):
		currentResultSet = []
		result = ((startX, startY, endX, endY), text)
		currentResultSet.append(result)
		groupedResults.append(currentResultSet)
		

# Now that we have grouped all the words, based on the line of text they belong to
# We'll order those words to form meaningful sentences

# 'newResults' is same as 'results', except, it contains full sentences instead of individual words
# All the sentences are translated as well
newResults = []
translator = Translator()
# Translation language. Make sure you have the font file in place for the language you are using
language = 'hi'
# path to the font file for the above language
fontpath = "../hi.ttf"
for resultSets in groupedResults:
	resultSets = sorted(resultSets, key=lambda r:r[0][0])
	x1,y1,x2,y2 = -1,-1,-1,-1
	combinedText = ""

	# for each word (seperated by a space) is combined to form a sentence
	# start x and y are taken from the first word
	# end x and y are taken from the last word
	for result in resultSets:
		if(x1 < 0):
			x1 = result[0][0]
			y1 = result[0][1]
		x2 = result[0][2]
		y2 = result[0][3]
		combinedText = combinedText+" "+result[1]

	# translate the sentence
	translation = translator.translate(combinedText, dest=language)
	transText = translation.text
	print(combinedText+' => '+transText )
	newResult = ((x1,y1,x2,y2),transText)
	newResults.append(newResult)

# @TODO : Can we figure out the color of the text from image too? 2nd most dominant color maybe?
# color values for the text
b,g,r,a = 0,0,0,0
# This is created to be able to write on image with the font of our choice
img_pil = Image.fromarray(orig.copy())
draw = ImageDraw.Draw(img_pil)

# ...................................................................................................

# loop over the results
for ((startX, startY, endX, endY), text) in newResults:
	output = orig.copy()
	cv2.rectangle(output, (startX, startY), (endX, endY),
		(0, 0, 255), 2)

	# Sharib ............................................................................................

	# Fetch the background color of text. We are assuming the most dominant color to be the background color
	# @TODO : This doesn't work in case of thick letters. Most dominant color is actually the text color in that case
	color = get_dominant_color(output[startY:startY+endY,startX:startX+endX])
	# fetch RGB values for background color
	backB,backG,backR = int(color[0]),int(color[1]),int(color[2])
	# margin by which the ROI will be increased.
	# This is to cover the parts of letter which sometimes gets clipped out by EAST
	# Nothing is perfect !
	# @TODO : This margin shouldn't be hardcoded
	margin = 5
	# Cover the actual text by drawing a filled retangle over it
	# Translated text will be written on top of this rectangle
	draw.rectangle([startX-margin, startY-margin, endX+margin, endY+margin], fill=(backB,backG,backR), outline=None)
	# A rough formulae to figure out the font height
	fontHeight = round((endY-startY)/1.2)
	font = ImageFont.truetype(fontpath, fontHeight)

	#try to center the text inside the rectangle by adding some offset to start X
	offset = 0
	length = len(text)
	# a rough formula to predict the length of translated text in pixels
	# works on the assumption that fontWidth(in pixels) = fontHeight/2
	# @TODO : This might not work if the screen resolution changes. Needs a better way of calculating the length.
	predictedLength = length * (fontHeight/2)
	availableLength = endX-startX
	if(predictedLength < availableLength):
		offset = int((availableLength-predictedLength)/2)
	# Write the translated text
	draw.text((startX+offset, startY),  text, font = font, fill = (b, g, r, a))
	img = np.array(img_pil)

	# show the output image
	cv2.imshow("Original", orig)
	cv2.imshow("Translated", img)

	# ...................................................................................................
cv2.waitKey(0)
