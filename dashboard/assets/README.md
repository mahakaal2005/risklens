# dashboard/assets/

No static assets exist here by design. This dashboard uses only Streamlit's built-in components (`st.title`, `st.metric`, `st.dataframe`, `st.container(border=True)`, emoji icons in string literals, etc.) — no external images, CDN-hosted fonts, icon libraries, or remote UI assets are loaded anywhere in `dashboard/`, so the app works fully offline once Python dependencies are installed.

If a local static asset is ever added here in a future milestone, it must be a file committed to this repository (never fetched from a URL at runtime) and referenced via a local relative path.
