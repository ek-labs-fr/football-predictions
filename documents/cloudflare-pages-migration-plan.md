# FPHostingStack → Cloudflare Pages Migration Plan

**Goal:** Host the Angular frontend on Cloudflare Pages with a custom domain (`ericfc.com`), proxy `/data/*` requests to the existing S3 data bucket, and unblock sharing the site with colleagues and friends without waiting on the AWS CloudFront verification ticket.

**Why this path:** Cloudflare Pages gives free TLS, global edge delivery, GitHub-connected CI/CD, and free custom-domain support — all without the AWS CloudFront verification queue. The trade-off (stack spans AWS + Cloudflare) is small since the prediction API and data bucket stay on AWS regardless.

---

## Architecture overview

```
                      ┌──────────────────────────────┐
   ericfc.com  ─────► │   Cloudflare Pages (Angular) │
                      └──────────────┬───────────────┘
                                     │
                  ┌──────────────────┼──────────────────┐
                  │                                     │
         /data/*  ▼                          /api/*     ▼
       ┌──────────────────┐            ┌────────────────────────┐
       │ Cloudflare Worker│            │ FastAPI prediction API │
       │  (proxy to S3)   │            │       (AWS)            │
       └────────┬─────────┘            └────────────────────────┘
                │
                ▼
       ┌──────────────────┐
       │ S3 data bucket   │
       │  (private)       │
       └──────────────────┘
```

Everything app-relative stays app-relative. The frontend doesn't know or care that `/data/*` is served by a Worker proxying S3 — which means hosting can move again later without frontend changes.

---

## Phase 1: Cloudflare Pages setup

### 1.1 Connect the repo

1. Sign in to the Cloudflare dashboard, go to **Workers & Pages** → **Create** → **Pages** → **Connect to Git**.
2. Authorize the GitHub app on the FPHostingStack repo.
3. Pick the branch to deploy from. For a semi-permanent share URL, use `main`. Pushes to `main` will auto-deploy; other branches will get preview URLs automatically.

### 1.2 Build configuration

| Setting | Value |
|---|---|
| Framework preset | Angular |
| Build command | `npm ci && npm run build -- --configuration production` |
| Build output directory | `dist/<project-name>/browser` (Angular 17+) or `dist/<project-name>` (older) |
| Root directory | (leave blank unless the Angular app is in a subfolder) |
| Node version | Set `NODE_VERSION` env var to match the local dev version (e.g., `20`) |

### 1.3 SPA fallback

Cloudflare Pages handles SPA routing automatically when it sees an Angular build, but verify by adding a `_redirects` file to the build output (or to the `public/` folder so it gets copied):

```
/*    /index.html   200
```

This sends every unmatched route to `index.html` so the Angular router can handle it client-side. Note: the Worker route for `/data/*` (Phase 2) takes precedence over this fallback.

---

## Phase 2: `/data/*` Worker proxy

### 2.1 Why a Worker (vs. direct fetch to S3)

- Keeps `/data/*` as a clean app-relative path — no frontend changes if hosting moves later.
- S3 data bucket can stay private; the Worker holds credentials.
- Single origin from the browser's perspective → no CORS preflight on data fetches.

### 2.2 Bucket access strategy

Two options, pick based on data sensitivity:

**Option A — Public-read prefix (simplest).** If the JSONs aren't sensitive, set a bucket policy allowing public read on the `/data/*` prefix only. Worker just does an unauthenticated `fetch()`. ~10 lines of code.

**Option B — Signed S3 requests (proper).** Worker holds AWS credentials in Cloudflare secrets and signs requests with SigV4. More code (~50 lines or use `aws4fetch`), but the bucket stays fully private.

Default recommendation: **Option A** unless the data is sensitive. The data bucket fronting a public website is already de-facto public; making it explicit is honest.

> **Operational constraint when Option A is in effect:** anything written under
> `web/data/*` is publicly readable on the open internet, both via the Pages
> Function (`ericfc.com/data/*`) and via the raw S3 URL. Only include fields
> that are safe to expose to anyone. Sensitive data (API tokens, PII,
> internal-only metrics) must live under a different bucket prefix.

### 2.3 Worker sketch (Option A)

```javascript
// functions/data/[[path]].js  — Pages Functions style
export async function onRequest(context) {
  const { request, params } = context;
  const path = Array.isArray(params.path) ? params.path.join('/') : params.path;
  const url = `https://<data-bucket>.s3.<region>.amazonaws.com/data/${path}`;

  const upstream = await fetch(url, {
    method: 'GET',
    headers: { 'Accept': request.headers.get('Accept') ?? '*/*' },
  });

  if (!upstream.ok) {
    return new Response(`Data fetch failed: ${upstream.status}`, { status: upstream.status });
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      'Content-Type': upstream.headers.get('Content-Type') ?? 'application/json',
      'Cache-Control': 'public, max-age=300',  // tune to data freshness needs
    },
  });
}
```

Drop this in `functions/data/[[path]].js` in the repo and Cloudflare Pages picks it up automatically — no separate Worker deployment needed.

### 2.4 Caching note

Set `Cache-Control` deliberately. If the JSONs update on every model retrain, `max-age=300` (5 min) is reasonable. If they're more static, bump it higher. Cloudflare's edge will cache aggressively, which is the whole point.

---

## Phase 3: Custom domain — `ericfc.com`

### 3.1 If `ericfc.com` is registered elsewhere

1. In Cloudflare dashboard: **Add a site** → enter `ericfc.com` → pick the Free plan.
2. Cloudflare scans existing DNS records and gives you two nameservers (e.g., `kate.ns.cloudflare.com`, `walt.ns.cloudflare.com`).
3. At the current registrar, change the nameservers to the two Cloudflare ones. Propagation: typically 1–24 hours.
4. Once Cloudflare shows the domain as **Active**, proceed to 3.3.

### 3.2 If `ericfc.com` is not yet registered

Register it through Cloudflare Registrar (at-cost pricing, no markup) — saves the nameserver step entirely. Or register anywhere and follow 3.1.

### 3.3 Attach the domain to the Pages project

1. In the Pages project → **Custom domains** → **Set up a custom domain**.
2. Add `ericfc.com` (apex) and `www.ericfc.com`.
3. Cloudflare creates the DNS records (CNAME flattening on the apex) and provisions a TLS certificate via Let's Encrypt or Google Trust Services. Usually live within minutes.
4. Set up an apex → www redirect (or vice versa) via a Page Rule or Bulk Redirect — pick one canonical and stick with it. Recommend `www.ericfc.com` → canonical, apex redirects to it (slightly easier on email/MX setups later).

### 3.4 TLS verification

Once the domain is active, hit `https://ericfc.com` and check:
- Certificate is valid and issued to `ericfc.com`.
- HTTP automatically redirects to HTTPS (Cloudflare default: **SSL/TLS** → set mode to **Full** or **Full (strict)** if the origin supports it; for Pages, **Full** is fine).

---

## Phase 4: Decommission FPHostingStack

Once `ericfc.com` is serving from Cloudflare Pages and verified working:

1. Keep the AWS CloudFront verification ticket open or close it — your call. Closing it is fine; you can always reopen later if you want to migrate back.
2. Tear down `FPHostingStack` via `cdk destroy FPHostingStack`. Leave the data bucket stack alone — Cloudflare is reading from it.
3. Update any internal docs / READMEs pointing at the old AWS URL.

---

## Branch strategy

Given the goal is a semi-permanent shared URL that will keep evolving:

- **`main`** — what's live at `ericfc.com`. Cloudflare Pages deploys this on every push.
- **Feature branches** — get automatic Cloudflare preview URLs (`<branch>.<project>.pages.dev`). Useful for sharing WIP with one or two people without touching the main URL.
- **No `develop` branch needed** for this scale. The preview URLs cover the "stage before sharing" use case that `develop` would otherwise serve.

If the project grows to multiple contributors or scheduled releases, revisit and add `develop` then.

---

## Checklist

- [ ] Connect GitHub repo to Cloudflare Pages
- [ ] Configure build command and output directory
- [ ] Add `_redirects` for SPA fallback
- [ ] Decide on bucket access strategy (public-read prefix vs. signed)
- [ ] Implement `functions/data/[[path]].js` Worker
- [ ] Test `/data/*` requests against a Pages preview URL
- [ ] Move `ericfc.com` DNS to Cloudflare (or register via Cloudflare)
- [ ] Attach `ericfc.com` and `www.ericfc.com` to the Pages project
- [ ] Verify TLS and HTTP→HTTPS redirect
- [ ] Configure apex/www canonical redirect
- [ ] Smoke-test the live site end-to-end
- [ ] `cdk destroy FPHostingStack`
- [ ] Share `https://ericfc.com` 🎉

---

## Open questions to resolve before starting

1. **Data sensitivity** — is `/data/*` content OK to be publicly readable from S3, or does it need signed access? Determines Phase 2.2 path.
2. **Angular version** — affects the build output directory in Phase 1.2 (`dist/<name>/browser` for v17+ vs. `dist/<name>` for older).
3. **Email on `ericfc.com`** — if you'll want email at this domain later, set up MX records before locking in the apex/www redirect direction.
