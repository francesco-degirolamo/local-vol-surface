# Security Guidelines

## 🔒 Critical Security Practices

### 1. Never Commit Credentials

The following files contain sensitive information and **must NEVER be committed to Git**:

- `wrds_config.py` - Contains WRDS username/password
- `.env` - Environment variables
- `.pgpass` - PostgreSQL credentials cache
- `.wrdsauth` - WRDS authentication cache

✅ All these files are already in `.gitignore`

### 2. Environment Variables

**RECOMMENDED**: Use environment variables instead of hardcoded credentials.

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
WRDS_USERNAME=your_actual_username
FLASK_DEBUG=False
```

Load in terminal:

```bash
export $(cat .env | xargs)
```

### 3. Production Deployment

**NEVER run in production with**:
- `FLASK_DEBUG=True` (enables code execution)
- Hardcoded passwords
- Default credentials

**ALWAYS**:
- Set `FLASK_DEBUG=False`
- Use environment variables
- Restrict network access (`host='127.0.0.1'` for local only)
- Use HTTPS if exposed to network
- Enable rate limiting

### 4. If Credentials Were Exposed

If you accidentally committed credentials to Git:

1. **Immediately change WRDS password** at https://wrds-www.wharton.upenn.edu/
2. Remove from Git history:
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch wrds_config.py" \
     --prune-empty --tag-name-filter cat -- --all
   ```
3. Force push (⚠️ destructive):
   ```bash
   git push origin --force --all
   ```
4. Notify your team/advisor

### 5. WRDS Authentication Best Practices

- Use **Duo Mobile** for 2FA
- Credentials are cached in `~/.pgpass` after first login
- Cache expires after 30 days
- Never share `~/.pgpass` file

### 6. Dependency Security

Keep dependencies updated:

```bash
pip install --upgrade -r requirements.txt
```

Check for vulnerabilities:

```bash
pip install safety
safety check
```

### 7. Code Review Checklist

Before committing:
- [ ] No hardcoded credentials
- [ ] `.env` not in repository
- [ ] `FLASK_DEBUG=False` in production code
- [ ] Sensitive files in `.gitignore`
- [ ] No API keys or tokens in code

---

**Last updated**: February 7, 2026
