# SpamShield
Description: College project to detect spam mails

How to run this project
1) Open terminal or command prompt.
  (Note: Make sure you are in the working folder {.../SpamShiled})

2) Create virtural space
For Linux User:
   mkdir env
   (run the command on the terminal or command prompt)

   python3 -m venv env
	   OR
   python -m venv env
   (run the command on the terminal or command prompt)

   source env/bin/activate
   (run the command on the terminal or command prompt)
  (Note: To deactive the virtual space, run the command : deactivate)
For Windows User:
    md env
    (run the command on the terminal or command prompt)

    python3 -m venv env
    (run the command on the terminal or command prompt)

    env\Scripts\activate
    (run the command on the terminal or command prompt)
  (Note: To deactive the virtual space, run the command : deactivate)

3) Install necessary requirements.
    pip install -r requirements.txt 
                OR

    (run the command on the terminal or command prompt)

4) Prepare the Model
There are two ways to do this. (you have to choose either a path or b path)
a) Run the command: python3 SpamShield_Model.py 
				OR
		    python SpamShield_Model.py
b) Install jupyter: pip install jupyterlab
   Run the jupyterlab: jupyter lab (jupyterlab interface will open) 
   Open the the model preparation file (SpamShield_Model.ipynb) now using the interface execute it.
(After this you will get two files: vectorizer.pkl and model.pkl {Note: if files are already exist, delete them first})

5)Finally, its time to RUN your program using streamlit
a)Install streamlit:
    pip install streamlit
            OR
    pip3 install streamlit
b)Run the program: streamlit run app.py in the terminal or command prompt. 
(Note: if you cannot be able to redirect to the http://localhost:8501.
       Open any web browser and copy-paste this url: http://localhost:8501 in the search bar.)
