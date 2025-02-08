# Modificación del html_template.py
css = '''
<style> 
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    background-color: #f8f9fa;
}
.chat-message.user {
    background-color: #e9ecef;
    border-left: 5px solid #2e7d32;  /* Verde oscuro para tema ambiental */
}
.chat-message.bot {
    background-color: #e9ecef;
    border-left: 5px solid #4caf50;  /* Verde para tema ambiental */
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 68px;
    max-height: 68px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #212529;
}
'''

logo = '''
<div style="margin-bottom: 15px; text-align: center;">
    <img src="https://www.policia.gov.co/sites/default/files/funcionalidades/comandos/proteccion-ambiental.png" 
         alt="Logo Policía Ambiental" style="max-width: 25%; height: auto;">
</div>
'''