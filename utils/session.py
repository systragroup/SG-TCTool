import uuid
from utils.data import DataManager

class SessionManager:
    """
    Manages user sessions for the traffic counting application.
    
    Handles creation, retrieval, updating, and deletion of session data,
    allowing multiple users to process different videos simultaneously.
    """
    
    def __init__(self):
        """Initialize session storage."""
        self.sessions = {}

    def create_session(self, session_id = None):
        """
        Create a new session with optional custom ID.
        
        Args:
            session_id: Optional custom session identifier
            
        Returns:
            str: The created session ID
        """
        if not session_id : session_id = str(uuid.uuid1())
        self.sessions[session_id] = {}
        self.sessions[session_id]['data_manager'] = DataManager()
        self.sessions[session_id]['progress'] = {}
        self.sessions[session_id]['results'] = {}
        return session_id

    def get_session_data(self, session_id):
        return self.sessions.get(session_id, None)

    def update_session_data(self, session_id, key, value):
        if session_id in self.sessions:
            self.sessions[session_id][key] = value

    def delete_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def clear_session(self, session_id):
        if session_id in self.sessions:
            self.sessions[session_id].clear()
