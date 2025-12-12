"""
LIBRAS Hands-Only System with ElevenLabs TTS
Captures ONLY HANDS for LIBRAS word recognition
Speaks recognized words in Portuguese using ElevenLabs

Requirements:
pip install mediapipe opencv-python numpy scikit-learn scipy elevenlabs pygame python-dotenv

Usage:
1. Run the script
2. Press 'r' to start recording a new WORD
3. Perform the sign with your hands
4. Press 'SPACE' when finished
5. Type what the word means in Portuguese
6. The system will recognize it by comparing KEYFRAMES
7. Recognized words are spoken in Portuguese!
8. Press 'q' to quit
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import threading
from io import BytesIO

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv não instalado. Instale: pip install python-dotenv")
    pass

# ElevenLabs TTS Integration
try:
    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs
    import pygame
    HAS_ELEVENLABS = True
except ImportError:
    print("⚠️  ElevenLabs não instalado. Instale: pip install elevenlabs pygame")
    HAS_ELEVENLABS = False


class ElevenLabsTTS:
    """Text-to-Speech usando ElevenLabs em Português"""
    def __init__(self, api_key=None):
        if not HAS_ELEVENLABS:
            self.enabled = False
            return
        
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        
        if not self.api_key:
            print("⚠️  ELEVENLABS_API_KEY não configurada")
            print("   Configure: export ELEVENLABS_API_KEY='sua_chave'")
            self.enabled = False
            return
        
        try:
            self.client = ElevenLabs(api_key=self.api_key)
            pygame.mixer.init()
            
            self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel (multilingual)
            self.voice_settings = VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True
            )
            
            self.enabled = True
            print("✓ ElevenLabs TTS configurado (Português)")
            
        except Exception as e:
            print(f"❌ Erro ao inicializar ElevenLabs: {e}")
            self.enabled = False
    
    def speak(self, text):
        """Fala o texto em Português (executa em thread separada)"""
        if not self.enabled:
            return
        
        thread = threading.Thread(target=self._speak_thread, args=(text,))
        thread.daemon = True
        thread.start()
    
    def _speak_thread(self, text):
        """Thread worker para síntese e reprodução de áudio"""
        try:
            # Use text_to_speech.convert() instead of generate()
            audio_generator = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_multilingual_v2",
                voice_settings=self.voice_settings
            )
            
            audio_data = b"".join(audio_generator)
            audio_stream = BytesIO(audio_data)
            
            pygame.mixer.music.load(audio_stream)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
        except Exception as e:
            print(f"❌ Erro ao falar '{text}': {e}")


class LibrasHandsOnlySystem:
    def __init__(self):
        # Initialize MediaPipe Hands ONLY
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,  # Very low for instant detection
            min_tracking_confidence=0.3,   # Very low for instant tracking
            model_complexity=0  # Use lite model for maximum speed
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        
        # Storage for learned WORDS
        self.words_db = {}
        self.load_words()
        
        # Recording state
        self.is_recording = False
        self.recorded_frames = []
        self.max_recording_frames = 600  # 20 seconds max
        
        # Recognition parameters (SINGLE FRAME MODE)
        ################################################################################
        self.similarity_threshold = 0.98 # High threshold for cosine similarity
        self.keyframe_count = 1  # Only one frame needed for recording
        
        # Recognition - NO BUFFER (instant single frame)
        self.last_recognized_word = None
        self.recognition_cooldown = 0
        self.frame_skip = 1  # Process every frame
        
        # ElevenLabs TTS
        self.tts = ElevenLabsTTS()
        
    def load_words(self):
        """Load previously learned words"""
        if os.path.exists('libras_hands_words.json'):
            with open('libras_hands_words.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                for word_name, word_data in data.items():
                    self.words_db[word_name] = {
                        'keyframes': [np.array(kf) for kf in word_data['keyframes']],
                        'duration': word_data['duration'],
                        'learned_at': word_data['learned_at'],
                        'times_recognized': word_data.get('times_recognized', 0)
                    }
            print(f"✓ Carregadas {len(self.words_db)} palavras do banco de dados")
    
    def save_words(self):
        """Save learned words"""
        data = {}
        for word_name, word_data in self.words_db.items():
            data[word_name] = {
                'keyframes': [kf.tolist() for kf in word_data['keyframes']],
                'duration': word_data['duration'],
                'learned_at': word_data['learned_at'],
                'times_recognized': word_data['times_recognized']
            }
        with open('libras_hands_words.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ Salvas {len(self.words_db)} palavras")
    
    def extract_hand_features(self, hand_landmarks_list):
        """
        ULTRA SIMPLE: Just raw positions (normalized)
        """
        features = []
        
        if not hand_landmarks_list:
            return None
        
        # Get all landmarks from all hands
        for hand_landmarks in hand_landmarks_list[:2]:  # Max 2 hands
            for landmark in hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
        
        # Pad if only 1 hand detected
        if len(hand_landmarks_list) < 2:
            features.extend([0] * (21 * 3))  # Pad with zeros
        
        # Normalize to unit vector
        features = np.array(features)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points (in degrees)"""
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def extract_keyframes(self, all_frames, num_keyframes=1):
        """
        SINGLE FRAME: Just take the middle frame for recording
        """
        if not all_frames or len(all_frames) == 0:
            return None
        
        num_frames = len(all_frames)
        
        if num_frames == 1:
            return all_frames
        
        # Take middle frame (most stable)
        middle_idx = num_frames // 2
        return [all_frames[middle_idx]]
    
    def compare_single_frames(self, frame1, frame2):
        """
        ULTRA SIMPLE: Just cosine similarity between two feature vectors
        """
        if frame1 is None or frame2 is None:
            return 0.0
        
        # Direct cosine similarity
        similarity = cosine_similarity(
            frame1.reshape(1, -1),
            frame2.reshape(1, -1)
        )[0][0]
        
        return similarity
    
    def extract_sequence_from_recording(self, recorded_frames):
        """Extract features from all recorded frames"""
        all_features = []
        
        for frame_data in recorded_frames:
            hand_landmarks = frame_data.get('hands')
            
            if hand_landmarks:
                features = self.extract_hand_features(hand_landmarks)
                if features is not None:
                    all_features.append(features)
        
        if not all_features:
            return None
        
        return all_features
    
    def recognize_word(self, current_frame):
        """Recognize word from SINGLE FRAME - instant pose recognition"""
        if current_frame is None:
            return None, 0.0
        
        if len(self.words_db) == 0:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        # Compare single frame against all learned words (SIMPLE)
        for word_name, word_data in self.words_db.items():
            stored_frame = word_data['keyframes'][0]  # Just one frame
            
            # Direct comparison
            similarity = self.compare_single_frames(current_frame, stored_frame)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = word_name
        
        # Simple threshold check
        if best_similarity >= self.similarity_threshold:
            self.words_db[best_match]['times_recognized'] += 1
            return best_match, best_similarity
        
        return None, best_similarity
    
    def learn_new_word(self, word_name, all_frames):
        """Learn new word from recorded frames"""
        duration_seconds = len(all_frames) / 30
        
        # Extract keyframes
        keyframes = self.extract_keyframes(all_frames, self.keyframe_count)
        
        if keyframes is None:
            return False
        
        self.words_db[word_name] = {
            'keyframes': keyframes,
            'duration': duration_seconds,
            'learned_at': datetime.now().isoformat(),
            'times_recognized': 0
        }
        
        self.save_words()
        print(f"✓ Palavra aprendida: '{word_name}' ({duration_seconds:.1f}s, {len(keyframes)} keyframes)")
        return True
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(2)  # Using /dev/video2
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "="*70)
        print("=== SISTEMA LIBRAS - SOMENTE MÃOS + ELEVENLABS ===")
        print("="*70)
        print("\nCaptura APENAS as mãos (dedos e gestos)")
        print("Modo POSE ESTÁTICA - Reconhece instantaneamente por frame único")
        
        # Show TTS status
        print("\n🔊 Síntese de Voz:")
        if self.tts.enabled:
            print("  ✓ ElevenLabs ativo - Palavras serão faladas em Português")
        else:
            print("  ⚠️  ElevenLabs desabilitado - Configure ELEVENLABS_API_KEY")
        
        print("\nComandos:")
        print("  'R'     - Começar gravar palavra")
        print("  'SPACE' - Parar gravação e salvar")
        print("  'Q'     - Sair")
        print(f"\n📚 Palavras conhecidas: {len(self.words_db)}")
        
        if len(self.words_db) > 0:
            print(f"   → {', '.join(list(self.words_db.keys())[:8])}")
        
        print("="*70 + "\n")
        
        recognized_word = None
        recognized_conf = 0.0
        recognition_display_frames = 0
        frame_count = 0  # For frame skipping
        current_word = None
        current_conf = 0.0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
            
            frame_count += 1
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with Hands ONLY (skip frames for speed unless recording)
            should_process = self.is_recording or (frame_count % self.frame_skip == 0)
            
            if should_process:
                hand_results = self.hands.process(rgb_frame)
            else:
                hand_results = None
            
            # Draw hand landmarks
            if hand_results and hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
            
            # Recording mode
            if self.is_recording:
                if hand_results and hand_results.multi_hand_landmarks:
                    frame_data = {
                        'hands': hand_results.multi_hand_landmarks
                    }
                    self.recorded_frames.append(frame_data)
                    
                    frames_recorded = len(self.recorded_frames)
                    time_recorded = frames_recorded / 30.0
                    
                    # Recording UI
                    cv2.circle(frame, (50, 50), 25, (0, 0, 255), -1)
                    cv2.circle(frame, (50, 50), 25, (255, 255, 255), 3)
                    
                    cv2.putText(frame, f"GRAVANDO: {time_recorded:.1f}s", (100, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                    cv2.putText(frame, f"{frames_recorded} frames", (100, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.putText(frame, "SPACE = parar e salvar", (50, 140),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    if frames_recorded >= self.max_recording_frames:
                        print("\n⚠️  Limite máximo atingido (20s)")
                        self.is_recording = False
            
            # Recognition mode (SINGLE FRAME - NO BUFFER)
            else:
                if hand_results and hand_results.multi_hand_landmarks:
                    # Extract features from THIS FRAME ONLY
                    current_features = self.extract_hand_features(hand_results.multi_hand_landmarks)
                    
                    if current_features is not None:
                        word, conf = self.recognize_word(current_features)
                        current_word = word
                        current_conf = conf
                        
                        if word and word != self.last_recognized_word and self.recognition_cooldown == 0:
                            recognized_word = word
                            recognized_conf = conf
                            recognition_display_frames = 60
                            self.last_recognized_word = word
                            self.recognition_cooldown = 15  # Small cooldown to prevent flickering
                            
                            print(f"✓ Reconhecido: '{word}' ({conf:.1%})")
                            
                            # 🔊 Speak the recognized word in Portuguese
                            if self.tts.enabled:
                                self.tts.speak(word)
                                print(f"  🔊 Falando: '{word}'")
            
            if self.recognition_cooldown > 0:
                self.recognition_cooldown -= 1
            
            # Display recognition
            if recognition_display_frames > 0 and recognized_word:
                box_h = 130
                box_y = frame.shape[0] - box_h - 30
                
                cv2.rectangle(frame, (30, box_y), (frame.shape[1] - 30, box_y + box_h),
                            (0, 180, 0), -1)
                cv2.rectangle(frame, (30, box_y), (frame.shape[1] - 30, box_y + box_h),
                            (0, 255, 0), 4)
                
                cv2.putText(frame, str(recognized_word), (50, box_y + 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
                
                cv2.putText(frame, f"{recognized_conf:.1%} confianca", (50, box_y + 110),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)
                
                recognition_display_frames -= 1
                
                if recognition_display_frames == 0:
                    self.last_recognized_word = None
            
            # Info panel
            cv2.rectangle(frame, (0, 0), (350, 140), (40, 40, 40), -1)
            
            cv2.putText(frame, f"Palavras: {len(self.words_db)}", (15, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(frame, "Modo: Pose Estatica", (15, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Show current similarity
            if not self.is_recording and current_word:
                similarity_color = (0, 255, 0) if current_conf >= self.similarity_threshold else (100, 100, 100)
                cv2.putText(frame, f"{current_word}: {current_conf:.2f}", (15, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, similarity_color, 2)
            
            if not self.is_recording:
                cv2.putText(frame, "R=gravar | Q=sair", (15, 125),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            
            cv2.imshow('LIBRAS Full Body System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('r') and not self.is_recording:
                self.is_recording = True
                self.recorded_frames = []
                print("\n🔴 Gravação iniciada! Mantenha a pose da mão...")
            
            elif key == ord(' ') and self.is_recording:
                self.is_recording = False
                
                if len(self.recorded_frames) < 10:
                    print("⚠️  Gravação muito curta!")
                    self.recorded_frames = []
                    continue
                
                all_features = self.extract_sequence_from_recording(self.recorded_frames)
                
                if all_features:
                    cv2.putText(frame, "Digite no terminal...", (50, 200),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                    cv2.imshow('LIBRAS Full Body System', frame)
                    cv2.waitKey(1)
                    
                    word_name = input(f"\n✏️  Palavra ({len(self.recorded_frames)} frames): ").strip()
                    
                    if word_name:
                        success = self.learn_new_word(word_name, all_features)
                        if not success:
                            print("❌ Erro ao salvar palavra")
                    else:
                        print("⚠️  Nome não fornecido")
                else:
                    print("❌ Não foi possível extrair features")
                
                self.recorded_frames = []
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        # Stats
        print("\n" + "="*70)
        print("📊 Estatísticas da Sessão:")
        print(f"   Total de palavras: {len(self.words_db)}")
        
        if self.words_db:
            total_rec = sum(w['times_recognized'] for w in self.words_db.values())
            print(f"   Reconhecimentos: {total_rec}")
            
            if total_rec > 0:
                most_rec = max(self.words_db.items(), key=lambda x: x[1]['times_recognized'])
                print(f"   Mais reconhecida: '{most_rec[0]}' ({most_rec[1]['times_recognized']}x)")
        
        print("="*70)

if __name__ == "__main__":
    system = LibrasHandsOnlySystem()
    system.run()