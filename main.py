# Imported necessary libraries
from tkinter import *
import tkinter as tk
import tkinter.messagebox as mbox
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
from fastcnn import DetectorAPI
from slidingwindow import AdvancedObjectDetectionApp

# Main Window & Configuration
window = tk.Tk()
window.title("Détection d'objets avec CNN et fenêtre glissante : comparaison avec Fast CNN")
window.geometry('1000x700')

# Top label
start1 = tk.Label(text="Détection d'objets avec CNN et fenêtre glissante :\nComparaison avec Fast CNN",
                  font=("Arial", 20), fg="black", justify="center")
start1.place(relx=0.5, y=80, anchor="center")

# Function defined to start the main application
def start_fun():
    window.destroy()

# Created a start button

# Image on the main window
img1 = ImageTk.PhotoImage(Image.open("img1.png"))
panel1 = tk.Label(window, image=img1)
panel1.place(relx=0.5, y=300, anchor="center")

# Image detection options
def image_option():
    # New window created for image section
    windowi = tk.Tk()
    windowi.title("Détection d'objets")
    windowi.geometry('1000x700')

    # Initialize variables
    global max_count1, max_acc1, max_avg_acc1
    max_count1 = 0
    max_acc1 = 0
    max_avg_acc1 = 0
    filename1 = ""

    # Function defined to open the image
    def open_img():
        global filename1
        filename1 = filedialog.askopenfilename(title="Sélectionner une image", parent=windowi)
        path_text1.delete("1.0", "end")
        path_text1.insert(END, filename1)

    # Function defined to detect the image
    def det_img():
        global filename1
        if not filename1:
            mbox.showerror("Error", "Aucun image sélectionné!", parent=windowi)
            return
        info1.config(text="Statut : Détection en cours...")
        mbox.showinfo("Statu", "Détection, veuillez patienter...", parent=windowi)
        detectByPathImage(filename1)

    # Main detection process here
    def detectByPathImage(path):
        global max_count1, max_acc1, max_avg_acc1

        odapi = DetectorAPI()
        threshold = 0.7
        image = cv2.imread(path)
        img = cv2.resize(image, (image.shape[1], image.shape[0]))
        boxes, scores, classes, num = odapi.processFrame(img)

        person_count = 0
        acc = 0

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                person_count += 1
                cv2.rectangle(img, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (255, 0, 0), 2)
                cv2.putText(img, f'P{person_count}, {round(scores[i], 2)}', (int(box[1]) - 30, int(box[0]) - 8),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                acc += scores[i]
                max_acc1 = max(max_acc1, scores[i])

        max_count1 = max(max_count1, person_count)
        if person_count > 0:
            max_avg_acc1 = max(max_avg_acc1, acc / person_count)

        cv2.imshow("Fast CNN", img)
        info1.config(text="Statut : Détection terminés")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # For images ----------------------
    lbl2 = tk.Label(windowi, text="Sélectionner une image", font=("Arial", 30), fg="blue")
    lbl2.place(x=80, y=200)
    path_text1 = tk.Text(windowi, height=1, width=37, font=("Arial", 30), bg="light yellow", fg="black", borderwidth=2, relief="solid")
    path_text1.place(x=80, y=260)

    Button(windowi, text="SÉLECTIONNER", command=open_img, cursor="hand2", font=("Arial", 20), bg="blue", fg="white").place(x=220, y=350)
    Button(windowi, text="Détecter", command=det_img, cursor="hand2", font=("Arial", 20), bg="blue", fg="white").place(x=620, y=350)

    info1 = tk.Label(windowi, font=("Arial", 30), fg="gray")
    info1.place(x=100, y=445)


# Button for Fast CNN option
Button(window, text="Fast CNN", command=image_option, font=("Arial", 25), bg="blue", fg="white", cursor="hand2", borderwidth=3, relief="raised").place(x=530, y=500)
Button(window, text="Fenêtre glissante", command=lambda: AdvancedObjectDetectionApp(window), font=("Arial", 25), bg="blue", fg="white", cursor="hand2", borderwidth=3, relief="raised").place(x=220, y=500)

window.mainloop()
