"""
Módulo para gestión automática de cache de FastEmbed y HuggingFace.
Configuración dinámica de rutas y manejo de directorios de cache.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CacheManager:
    """Gestor de cache para modelos de embeddings y LLM."""
    
    def __init__(self):
        self.cache_info: Dict[str, Any] = {}
        self.fastembed_cache_dir: str = ""
        self.hf_cache_dir: str = ""
        
    def setup_cache_directories(self) -> Dict[str, Any]:
        """
        Configura automáticamente los directorios de cache con rutas dinámicas.
        
        Returns:
            Dict con información de configuración del cache
        """
        try:
            # Detectar directorio home dinámicamente
            user_home = Path.home()
            cache_base = user_home / ".cache"
            
            # Crear estructura de directorios
            fastembed_cache = cache_base / "fastembed"
            hf_cache = cache_base / "huggingface"
            
            # Crear directorios con permisos seguros
            for directory in [cache_base, fastembed_cache, hf_cache]:
                directory.mkdir(mode=0o755, parents=True, exist_ok=True)
            
            # Configurar variables de entorno ANTES de importar bibliotecas
            os.environ["FASTEMBED_CACHE_PATH"] = str(fastembed_cache)
            os.environ["HF_HOME"] = str(hf_cache)
            
            # Verificar espacio en disco disponible (mínimo 3GB)
            import shutil
            free_space_gb = shutil.disk_usage(user_home).free / (1024**3)
            
            # Guardar información de cache
            self.fastembed_cache_dir = str(fastembed_cache)
            self.hf_cache_dir = str(hf_cache)
            
            self.cache_info = {
                "fastembed_cache": str(fastembed_cache),
                "hf_cache": str(hf_cache),
                "user_home": str(user_home),
                "user_name": user_home.name,
                "free_space_gb": round(free_space_gb, 2),
                "success": True,
                "cache_base": str(cache_base)
            }
            
            logger.info("✅ Cache configurado automáticamente:")
            logger.info(f"👤 Usuario: {self.cache_info['user_name']}")
            logger.info(f"🏠 Home: {self.cache_info['user_home']}")
            logger.info(f"📁 FastEmbed: {self.cache_info['fastembed_cache']}")
            logger.info(f"📁 HuggingFace: {self.cache_info['hf_cache']}")
            logger.info(f"💾 Espacio libre: {self.cache_info['free_space_gb']} GB")
            
            if self.cache_info['free_space_gb'] < 3:
                logger.warning("⚠️ Espacio en disco bajo (<3GB). Los modelos podrían no descargarse correctamente.")
                
            return self.cache_info
            
        except Exception as e:
            return self._setup_fallback_cache(e)
    
    def _setup_fallback_cache(self, error: Exception) -> Dict[str, Any]:
        """
        Configura cache de respaldo en caso de error.
        
        Args:
            error: Excepción que causó el fallback
            
        Returns:
            Dict con información de cache de respaldo
        """
        logger.warning(f"⚠️ Error configurando cache principal: {error}")
        logger.warning("🔄 Configurando cache de respaldo...")
        
        # Fallback seguro a directorios temporales
        temp_base = Path("/tmp")
        temp_fastembed = temp_base / f"fastembed_cache_{os.getpid()}"
        temp_hf = temp_base / f"hf_cache_{os.getpid()}"
        
        temp_fastembed.mkdir(exist_ok=True)
        temp_hf.mkdir(exist_ok=True)
        
        os.environ["FASTEMBED_CACHE_PATH"] = str(temp_fastembed)
        os.environ["HF_HOME"] = str(temp_hf)
        
        self.fastembed_cache_dir = str(temp_fastembed)
        self.hf_cache_dir = str(temp_hf)
        
        self.cache_info = {
            "fastembed_cache": str(temp_fastembed),
            "hf_cache": str(temp_hf),
            "user_home": "FALLBACK_/tmp",
            "user_name": "unknown",
            "free_space_gb": 0,
            "success": False,
            "error": str(error),
            "cache_base": str(temp_base)
        }
        
        logger.warning("⚠️ Cache configurado en modo fallback:")
        logger.warning(f"📁 FastEmbed: {self.cache_info['fastembed_cache']}")
        logger.warning(f"📁 HuggingFace: {self.cache_info['hf_cache']}")
        logger.warning(f"❌ Error original: {error}")
        
        return self.cache_info
    
    def create_embedding_model(self, config: Dict[str, Any]):
        """
        Crea el modelo de embeddings con verificación de cache.
        
        Args:
            config: Configuración de la aplicación
            
        Returns:
            Modelo de embeddings configurado
        """
        try:
            logger.info("🔄 Inicializando modelo de embeddings...")
            
            # Verificar si el modelo ya está en cache
            fastembed_cache_path = Path(self.fastembed_cache_dir)
            model_cache_path = fastembed_cache_path / "intfloat"
            
            if model_cache_path.exists() and any(model_cache_path.iterdir()):
                logger.info("✅ Modelo encontrado en cache local - carga rápida")
            else:
                logger.info("📥 Primera ejecución - descargando modelo (~2.24GB)")
                logger.info("⏳ Esto puede tardar varios minutos según tu conexión...")
                
            from llama_index.embeddings.fastembed import FastEmbedEmbedding
            
            # Configurar la variable de entorno para que RAG use nuestro cache
            os.environ["FASTEMBED_CACHE_PATH"] = self.fastembed_cache_dir
            
            embed_model = FastEmbedEmbedding(
                model_name=config["embedding_model"],
                cache_dir=self.fastembed_cache_dir
            )
            
            logger.info("✅ Modelo de embeddings cargado correctamente")
            
            # Verificar tamaño del cache después de la carga
            try:
                cache_size_mb = sum(f.stat().st_size for f in fastembed_cache_path.rglob('*') if f.is_file()) / (1024*1024)
                logger.info(f"📊 Tamaño del cache: {cache_size_mb:.1f} MB")
            except Exception:
                pass
            
            # Guardar referencia global para que RAG pueda usarla
            self._cached_embedding_model = embed_model
            
            return embed_model
            
        except Exception as e:
            logger.error(f"❌ Error al cargar modelo de embeddings: {e}")
            raise

    def get_cached_embedding_model(self):
        """
        Obtiene el modelo de embeddings cacheado.
        
        Returns:
            Modelo de embeddings si existe, None si no
        """
        return getattr(self, '_cached_embedding_model', None)

    def ensure_embedding_model_ready(self, config: Dict[str, Any]):
        """
        Asegura que el modelo de embeddings esté listo y configurado.
        
        Args:
            config: Configuración de la aplicación
        """
        # Configurar variables de entorno para que FastEmbed use nuestro cache
        os.environ["FASTEMBED_CACHE_PATH"] = self.fastembed_cache_dir
        os.environ["HF_HOME"] = self.hf_cache_dir
        
        # Pre-cargar el modelo si no está cacheado
        if not hasattr(self, '_cached_embedding_model') or self._cached_embedding_model is None:
            self.create_embedding_model(config)
        
        logger.info(f"✅ Variables de entorno configuradas para cache:")
        logger.info(f"   FASTEMBED_CACHE_PATH={os.environ.get('FASTEMBED_CACHE_PATH')}")
        logger.info(f"   HF_HOME={os.environ.get('HF_HOME')}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Obtiene información detallada del cache.
        
        Returns:
            Dict con información completa del cache
        """
        fastembed_path = Path(self.fastembed_cache_dir)
        hf_path = Path(self.hf_cache_dir)
        
        try:
            fastembed_size_mb = sum(f.stat().st_size for f in fastembed_path.rglob('*') if f.is_file()) / (1024*1024) if fastembed_path.exists() else 0
            hf_size_mb = sum(f.stat().st_size for f in hf_path.rglob('*') if f.is_file()) / (1024*1024) if hf_path.exists() else 0
        except Exception:
            fastembed_size_mb = 0
            hf_size_mb = 0
        
        return {
            "fastembed": {
                "path": str(fastembed_path),
                "exists": fastembed_path.exists(),
                "size_mb": round(fastembed_size_mb, 2)
            },
            "huggingface": {
                "path": str(hf_path),
                "exists": hf_path.exists(),
                "size_mb": round(hf_size_mb, 2)
            },
            "user_info": {
                "home": self.cache_info.get("user_home", "unknown"),
                "name": self.cache_info.get("user_name", "unknown"),
                "free_space_gb": self.cache_info.get("free_space_gb", 0)
            },
            "cache_status": self.cache_info.get("success", False),
            "total_cache_size_mb": round(fastembed_size_mb + hf_size_mb, 2)
        }
    
    def clear_cache(self) -> Dict[str, str]:
        """
        Limpia los directorios de cache.
        
        Returns:
            Dict con resultado de la operación
        """
        try:
            import shutil
            
            fastembed_path = Path(self.fastembed_cache_dir)
            hf_path = Path(self.hf_cache_dir)
            
            removed_files = 0
            if fastembed_path.exists():
                shutil.rmtree(fastembed_path)
                removed_files += 1
                logger.info(f"🗑️ Cache FastEmbed eliminado: {fastembed_path}")
                
            if hf_path.exists():
                shutil.rmtree(hf_path)
                removed_files += 1
                logger.info(f"🗑️ Cache HuggingFace eliminado: {hf_path}")
            
            # Recrear directorios vacíos
            fastembed_path.mkdir(parents=True, exist_ok=True)
            hf_path.mkdir(parents=True, exist_ok=True)
            
            return {
                "message": f"Cache limpiado correctamente. {removed_files} directorios eliminados.",
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ Error al limpiar cache: {e}")
            return {
                "message": f"Error al limpiar cache: {str(e)}",
                "status": "error"
            }
    
    def validate_cache_health(self) -> Dict[str, Any]:
        """
        Valida la salud del sistema de cache.
        
        Returns:
            Dict con información de salud del cache
        """
        fastembed_path = Path(self.fastembed_cache_dir)
        hf_path = Path(self.hf_cache_dir)
        
        health_info = {
            "overall_health": "healthy",
            "issues": [],
            "recommendations": []
        }
        
        # Verificar existencia de directorios
        if not fastembed_path.exists():
            health_info["issues"].append("Directorio FastEmbed no existe")
            health_info["overall_health"] = "warning"
            
        if not hf_path.exists():
            health_info["issues"].append("Directorio HuggingFace no existe")
            health_info["overall_health"] = "warning"
        
        # Verificar permisos
        try:
            test_file = fastembed_path / "test_write"
            test_file.touch()
            test_file.unlink()
        except Exception:
            health_info["issues"].append("Sin permisos de escritura en cache FastEmbed")
            health_info["overall_health"] = "error"
        
        # Verificar espacio en disco
        if self.cache_info.get("free_space_gb", 0) < 1:
            health_info["issues"].append("Espacio en disco muy bajo (<1GB)")
            health_info["overall_health"] = "error"
            health_info["recommendations"].append("Liberar espacio en disco")
        elif self.cache_info.get("free_space_gb", 0) < 3:
            health_info["issues"].append("Espacio en disco bajo (<3GB)")
            health_info["overall_health"] = "warning"
            health_info["recommendations"].append("Considerar liberar espacio en disco")
        
        return health_info


# Instancia global del gestor de cache
cache_manager = CacheManager()

def initialize_cache() -> Dict[str, Any]:
    """
    Función de conveniencia para inicializar el cache.
    
    Returns:
        Información de configuración del cache
    """
    print("🔧 Configurando cache automáticamente...")
    cache_info = cache_manager.setup_cache_directories()
    return cache_info

def get_cache_manager() -> CacheManager:
    """
    Obtiene la instancia del gestor de cache.
    
    Returns:
        Instancia de CacheManager
    """
    return cache_manager