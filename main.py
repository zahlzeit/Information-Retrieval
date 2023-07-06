from tkinter import *
from tkinter import ttk
from categorizer import *
# from tkinter.scrolledtext import ScrolledText

def resizeImage(img, newWidth, newHeight):
    oldWidth = img.width()
    oldHeight = img.height()
    newPhotoImage = PhotoImage(width=newWidth, height=newHeight)
    for x in range(newWidth):
        for y in range(newHeight):
            xOld = int(x*oldWidth/newWidth)
            yOld = int(y*oldHeight/newHeight)
            rgb = '#%02x%02x%02x' % img.get(xOld, yOld)
            newPhotoImage.put(rgb, (x, y))
    return newPhotoImage

root = Tk()

root.geometry('700x500+100+100')
root.title("CUSE && CUTC")
root.iconbitmap('D:\Python\IR-Assignment\Assignment\icons\Grad Hat.ico')
root.config(bg='#c5dff8')
root.resizable(0,0)

# Notebook is required to create tabs
my_notebook = ttk.Notebook(root)
my_notebook.pack(pady=10)

#Frames was created for Classifier
my_frame2 = Frame(my_notebook, width=700, height=500, bg = '#A0BFE0')
my_frame2.grid(row=0, column=0, sticky="NESW")
my_frame2.grid_rowconfigure(0, weight=1)
my_frame2.grid_columnconfigure(0, weight=1)

my_notebook.add(my_frame2, text = "Text Classification")

# This section is for the classifier

#This is the first row
tcLabel = Label(my_frame2, text = 'CU Text Classifier', font=('arial', 16, 'bold'), bg='#A0BFE0')
# tcLabel.grid(padx=5, pady=5, row=0, column=1)
tcLabel.grid(row=0, column=0)

#This is the second row
tcLabel1 = Label(my_frame2, text = 'Enter Your Text Below and Press Classify Button', font=('arial', 12, 'bold'), bg='#A0BFE0')
# tcLabel.grid(padx=5, pady=5, row=1, column=1)
tcLabel1.grid(row=1, column=0)

#This is the third row
textEntryBox = Entry(my_frame2, width = 45, font=('arial', 12), bd = 4, relief=SUNKEN)
textEntryBox.grid(row=2, column=0, pady=5)

#This is the fourth row
def buttonClassify():
    text = textEntryBox.get()
    print(text)
    if text == "":
        resultLabel.config(text=" Results: " + "Enter any phrase")
    
    else:
        textclass = classify(text)
        print(textclass)
        resultLabel.config(text=" Results: " + textclass)
    # resultLabel.grid(row=4, column=0)

classifyButton = Button(my_frame2, text="Classify", bg='#C5DFF8', font=('arial', 12), bd=1, cursor='hand2', command=buttonClassify)
classifyButton.grid(row=3, column=0)
resultLabel = Label(my_frame2, text=" Results: ", bg='#A0BFE0', font=('arial', 12))
resultLabel.grid(row=4, column=0)


root.mainloop()