from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
import base64

# Generate a new ECDSA P-256 key
private_key = ec.generate_private_key(ec.SECP256R1())
public_key = private_key.public_key()

# Export private key (PEM format)
private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

# Export public key in PEM
public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

# Export public key in URL-safe base64 (for the browser)
public_numbers = public_key.public_numbers()
x = public_numbers.x.to_bytes(32, 'big')
y = public_numbers.y.to_bytes(32, 'big')
public_key_bytes = b'\x04' + x + y
public_key_b64 = base64.urlsafe_b64encode(public_key_bytes).rstrip(b'=').decode('utf-8')

print("----- PRIVATE KEY (PEM) -----")
print(private_pem.decode('utf-8'))
print("----- PUBLIC KEY (PEM) -----")
print(public_pem.decode('utf-8'))
print("----- PUBLIC KEY (Base64 for Browser) -----")
print(public_key_b64)
