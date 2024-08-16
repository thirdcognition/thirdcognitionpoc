import re
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from lib.chain import get_chain
from lib.db_tools import FileDataTable, init_db
from lib.document_parse import markdown_to_text

with open('admin_auth.yaml') as file:
    auth_config = yaml.load(file, Loader=SafeLoader)

# authenticator: stauth.Authenticate = None
# = stauth.Authenticate(
#     auth_config['credentials'],
#     auth_config['cookie']['name'],
#     auth_config['cookie']['key'],
#     auth_config['cookie']['expiry_days'],
#     auth_config['pre-authorized']
# )

def check_auth():
    # global authenticator
    # if authenticator is None:
    authenticator = stauth.Authenticate(
        auth_config['credentials'],
        auth_config['cookie']['name'],
        auth_config['cookie']['key'],
        auth_config['cookie']['expiry_days'],
        auth_config['pre-authorized']
    )

    authenticator.login()
    if st.session_state['authentication_status']:
        col1, col2 = st.columns([7, 1])
        col1.write(f'Welcome *{st.session_state["name"]}*')
        with col2: authenticator.logout()
        return True
    else:
        if st.session_state['authentication_status'] is False:
            st.error('Username/password is incorrect')
        if st.session_state['authentication_status'] is None:
            st.warning('Please enter your username and password')

        return False

def get_all_categories():
    database_session = init_db()

    if "file_categories" not in st.session_state:
        uniq_categories = []
        categories = database_session.query(FileDataTable.category_tag).distinct()
        for items in categories:
            for category_list in items:
                if len(category_list) > 0:
                    for category in category_list:
                        if category not in uniq_categories:
                            uniq_categories.append(category)

        st.session_state.file_categories = uniq_categories
    else:
        uniq_categories = st.session_state.file_categories

    return uniq_categories

def llm_edit(chain, texts, guidance=None, force=False) -> tuple[str, str]:
    text = ""
    thoughts = ""

    i = 0
    total = len(texts)

    if total > 1 and chain == "summary":
        total += 1

    if not force and (texts == None or texts[0] == None or total == 1 and len(texts[0]) < 1000):
        return None, None

    bar = st.progress(0, text="Processing...")

    if total > 1:
        inputs = []
        for sub_text in texts:
            bar.progress(i / total, "Processing...")
            i += 1
            _text = re.sub(r"[pP]age [0-9]+:", "", sub_text)
            _text = re.sub(r"[iI]mage [0-9]+:", "", _text)
            input = {"context": _text}

            guided_llm = ""
            if guidance is not None and guidance != "":
                input["question"] = guidance
                guided_llm = "_guided"

            inputs.append(input)

            mid_results, mid_thoughts = get_chain(chain + guided_llm).invoke(input)

            text += mid_results + "\n\n"
            thoughts += mid_thoughts + "\n\n"

    else:
        _text = re.sub(r"[pP]age [0-9]+", "", texts[0])
        _text = re.sub(r"[iI]mage [0-9]+", "", _text)
        text = _text

    bar.progress((total - 1) / total, text="Processing...")

    if chain == "summary":
        text = markdown_to_text(text)

        input = {"context": text}

        guided_llm = ""

        if guidance is not None and guidance != "":
            guided_llm = "_guided"
            input["question"] = guidance

        text, thoughts = get_chain(chain + guided_llm).invoke(input)

    bar.empty()

    return text, thoughts