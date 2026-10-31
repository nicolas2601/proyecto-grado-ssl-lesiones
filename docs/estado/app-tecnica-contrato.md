# Sesion: app-tecnica-contrato  |  2026-10-30

## Tarea

Auditar el contrato tecnico entre la app desplegada en https://skin.tikno.pro y el servicio de inferencia desplegado, y
proponer los cambios pertinentes, sin tocar codigo.

## Contexto minimo (no leer mas que esto)

La app esta publicada en `https://skin.tikno.pro`, que sirve el frontend y expone el mismo backend que `https://derm.nicolasmoreno.site` (verificado: identico `model_sha256`). El cliente consume la API a traves de un proxy de Vite
sobre `/api`. El servicio corre el release V9 y hoy sirve por defecto el brazo `B71`, aunque
tambien expone `A71`. El servicio esta versionado en la rama `phone-web` del repositorio
`proyecto-grado-ssl-lesiones`, bajo `demo/backend/`.

Existe una interfaz alternativa, `demo/phone/`, en esa misma rama, que cumple una funcion
parecida a la de `Perma`. Determinar si se solapan es parte de la tarea.

## Entra

- `C:/Users/mpaul/Documents/skin-model-pwa/src/features/analysis/prediction.ts`
- `C:/Users/mpaul/Documents/skin-model-pwa/src/app/useConnectivity.ts`
- `C:/Users/mpaul/Documents/skin-model-pwa/vite.config.ts`
- Rama `origin/phone-web` del repo `ProyectoDeGrado`: `demo/backend/` y `demo/phone/`
  (leer con `git show origin/phone-web:<ruta>`, sin hacer checkout)
- `https://derm.nicolasmoreno.site/docs` y `/api/status` (solo GET)

## Sale

Un unico archivo: `docs/estado/app-tecnica-contrato.reporte.md`, con esta estructura:

1. **Contrato actual**: cada endpoint, su entrada, su salida y quien lo llama.
2. **Divergencias**: donde el cliente y el servidor asumen cosas distintas.
3. **`demo/phone/` contra `Perma`**: que hace cada una, en que se solapan, cual conviene
   conservar y por que.
4. **Propuestas**, cada una con cambio concreto, justificacion y costo (bajo / medio / alto).
5. **Trazabilidad**: que se puede y que no se puede acreditar hoy sobre el modelo en produccion.

## Criterios (binarios)

1. Cada afirmacion sobre el servicio desplegado se respalda con una respuesta HTTP real incluida
   en el reporte.
2. Se documenta el manejo de error del cliente para: 429, 5xx, timeout y sin conexion.
3. Se verifica si el cliente fija el modelo (`A71` / `B71`) o lo hereda del servidor, y si el
   usuario puede saber cual se uso.
4. Se revisa el comportamiento del service worker frente a `/api`, incluida la regla
   `navigateFallbackDenylist`.
5. Se emite un veredicto explicito sobre si `demo/phone/` y `Perma` se solapan.
6. El reporte distingue lo verificado de lo no verificado.

## Fuera de alcance

- No escribir ni modificar codigo, ni en este repo ni en `skin-model-pwa`.
- No hacer `checkout`, `merge`, `commit` ni `push` de ninguna rama.
- No llamar a `POST /api/predict` con imagenes de personas. Si se necesita probar inferencia,
  usar unicamente las imagenes golden ya versionadas.
- No opinar sobre la calidad cientifica del modelo ni sobre que brazo deberia servirse: eso lo
  decide la sesion maestra. Si, dejar registrado que el servicio expone los dos.
- No tocar configuracion de despliegue, DNS ni Cloudflare.

---
## Reporte  (lo completa la sesion al terminar)
Verificado: <con que comando>
No verificado: <y por que>
Bloqueos:
