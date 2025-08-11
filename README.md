# 📧 SpamShield
Description: Project to detect spam mails

--- 

🚀 How to run this project <br>
1️⃣ Open terminal or command prompt.<br>
  (Note: Make sure you are in the working folder {.../SpamShiled})<br>

---

2️⃣  Create virtural space<br>
💻 For Linux User:<br>
   mkdir env
   (run the command on the terminal or command prompt)

   python3 -m venv env
	   OR
   python -m venv env
   (run the command on the terminal or command prompt)

   source env/bin/activate
   (run the command on the terminal or command prompt)
  (Note: To deactive the virtual space, run the command : deactivate)<br>
🖥 For Windows User:<br>
    md env
    (run the command on the terminal or command prompt)

    python3 -m venv env
    (run the command on the terminal or command prompt)

    env\Scripts\activate
    (run the command on the terminal or command prompt)
  (Note: To deactive the virtual space, run the command : deactivate)
<br>

---

3️⃣  Install necessary requirements.<br>
    pip install -r requirements.txt 
                OR

    (run the command on the terminal or command prompt)
<br>

---

4️⃣ Prepare the Model 🛠 <br>
There are two ways to do this. (you have to choose either a path or b path)<br>
a) Run the command: python3 SpamShield_Model.py <br>
				OR <br>
		    python SpamShield_Model.py<br>
b) Install jupyter: pip install jupyterlab<br>
   Run the jupyterlab: jupyter lab (jupyterlab interface will open) <br>
   Open the the model preparation file (SpamShield_Model.ipynb) now using the interface execute it. <br>
(After this you will get two files: vectorizer.pkl and model.pkl {Note: if files are already exist, delete them first}) 
<br>

---

5️⃣ Finally, its time to RUN your program using streamlit 🎯  <br>
a)Install streamlit:  <br>
    pip install streamlit  <br>
            OR
    pip3 install streamlit  <br>
b)Run the program:
```streamlit run app.py ``` in the terminal or command prompt. <br>
(Note: if you cannot be able to redirect to the http://localhost:8501.  <br>
       Open any web browser and copy-paste this url: http://localhost:8501 in the search bar.)  <br>
