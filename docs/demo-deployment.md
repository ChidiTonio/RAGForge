# RAGForge — Stage 1 Demo Deployment

Date: 2026-10-16

## Status
Password-protected public demo is working and ready for external testing.
This is a demo deployment, not a production deployment.

## Architecture
Browser → Cloudflare Tunnel → Caddy → Streamlit

- Streamlit runs locally on 127.0.0.1:8501.
- Caddy provides Basic Authentication on port 8502.
- Cloudflare Tunnel provides temporary public HTTPS access.
- The tunnel uses HTTP/2.

## Verified
- Local Streamlit health endpoint returned HTTP 200.
- Public application loaded successfully.
- Initial page load was reported as under two seconds in one test.
- This was not a formal performance or load test.

## Current Limitations
- The computer must remain awake and connected to the internet.
- WSL, Streamlit, Caddy, and Cloudflare must remain running.
- Restarting the temporary tunnel may generate a new public URL.
- Streamlit browser.serverAddress must match the public hostname.
- Production security and multi-user capacity are not yet validated.

## Restart Checklist
1. Start Cloudflare Tunnel and copy the new public hostname.
2. Start Streamlit with the current public hostname configured.
3. Start Caddy using the existing working configuration.
4. Open the public URL and verify authenticated access.

## Security
Private demo files are stored in .local-demo/ and excluded from Git.
Never commit passwords, password hashes, API keys, or private documents.
Ignoring a directory does not remove files already tracked by Git.

## Next Steps
- Evaluate answer accuracy and relevance.
- Check retrieval quality and source support.
- Test document upload and chat functionality.
- Measure answer-generation latency.
- Test reliability across repeated sessions.
- Make startup repeatable.
- Plan persistent hosting and a stable public address.
