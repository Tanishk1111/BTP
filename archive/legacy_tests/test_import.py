import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'spatx_core'))

print("Testing import...")
from spatx_core.models.cit_to_gene.CiT_Net_T import CIT
print("✅ Import successful!")