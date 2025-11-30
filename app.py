import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    import uvicorn
    import os
    from app.routes import app

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

