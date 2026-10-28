"""Conversation support for the Google Generative AI Conversation integration."""

from __future__ import annotations

import asyncio
import base64
import codecs
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any, Literal, override
import wave

import aiohttp
import voluptuous as vol
import probatio

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_PROMPT, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_CHAT_MODEL,
    CONF_TEMPERATURE,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_TEMPERATURE,
)
from .entity import GoogleGenerativeAILLMBaseEntity

SUPPORTED_BIDI_SCHEMA_KEYS = {
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "items",
    "properties",
    "required",
}


def _escape_decode(value: Any) -> Any:
    """Recursively call codecs.escape_decode on all values."""
    if isinstance(value, str):
        return codecs.escape_decode(bytes(value, "utf-8"))[0].decode("utf-8")  # type: ignore[attr-defined]
    if isinstance(value, list):
        return [_escape_decode(item) for item in value]
    if isinstance(value, dict):
        return {k: _escape_decode(v) for k, v in value.items()}
    return value


def _format_bidi_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Format OpenAPI schema to Gemini BidiGenerateContent JSON schema."""
    if (subschemas := schema.get("anyOf")) or (subschemas := schema.get("allOf")):
        merged: dict[str, Any] = {}
        merged_props: dict[str, Any] = {}
        merged_req: list[str] = []
        chosen_branch: dict[str, Any] | None = None
        for subschema in subschemas:
            if not isinstance(subschema, dict):
                continue
            if "properties" in subschema or subschema.get("type") == "object":
                merged["type"] = "object"
                merged_props.update(subschema.get("properties", {}))
            if "required" in subschema and isinstance(subschema["required"], list):
                for r in subschema["required"]:
                    if r not in merged_req:
                        merged_req.append(r)
            if chosen_branch is None and "type" in subschema:
                chosen_branch = subschema
        if merged_props:
            merged["type"] = "object"
            merged["properties"] = merged_props
            if merged_req:
                merged["required"] = [r for r in merged_req if r in merged_props]
            return _format_bidi_schema(merged)
        if chosen_branch is not None:
            return _format_bidi_schema(chosen_branch)
        return _format_bidi_schema(subschemas[0])

    result: dict[str, Any] = {}
    for key, val in schema.items():
        if key not in SUPPORTED_BIDI_SCHEMA_KEYS:
            continue
        if key == "type":
            val = str(val).upper()
        elif key == "format":
            if schema.get("type") == "string" and val != "enum":
                continue
            if schema.get("type") not in ("number", "integer", "string"):
                continue
        elif key == "items":
            val = _format_bidi_schema(val) if isinstance(val, dict) else {"type": "STRING"}
        elif key == "properties":
            val = {k: _format_bidi_schema(v) for k, v in val.items() if isinstance(v, dict)}
        result[key] = val

    if "properties" in result and "type" not in result:
        result["type"] = "OBJECT"

    if result.get("enum") and result.get("type") != "STRING":
        result["type"] = "STRING"
        result["enum"] = [str(item) for item in result["enum"]]

    if result.get("type") != "OBJECT":
        result.pop("required", None)
        result.pop("properties", None)
        if "type" not in result:
            result["type"] = "STRING"
    else:
        if not result.get("properties"):
            result["properties"] = {"json": {"type": "STRING"}}
            result.pop("required", None)
        elif "required" in result:
            valid_req = [k for k in result["required"] if k in result["properties"]]
            if valid_req:
                result["required"] = valid_req
            else:
                result.pop("required", None)
    return result


def _format_bidi_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format HA LLM tool into a Gemini BidiGenerateContent functionDeclaration."""
    if tool.parameters.schema:
        parameters = _format_bidi_schema(
            probatio.to_openapi(tool.parameters, custom_serializer=custom_serializer)
        )
        if parameters.get("type") != "OBJECT":
            parameters = {
                "type": "OBJECT",
                "properties": {"value": parameters},
            }
    else:
        parameters = None

    decl: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description or tool.name,
    }
    if parameters:
        decl["parameters"] = parameters
    return decl


_BARE_AREA_LIGHT_MAP: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...] = (
    (
        ("v obýváku", "v obyvaku", "obývák", "obyvak", "v obývacím pokoji", "obývací pokoj"),
        "v obýváku",
        ("light.ovladani_osvetleni_kuchyne",),
    ),
    (
        ("v kuchyni", "kuchyň", "kuchyne", "kuchyni", "kuchyně"),
        "v kuchyni",
        ("light.ovladani_osvetleni_kuchyne",),
    ),
    (
        ("v zádveří", "v zadveri", "zádveří", "zadveri"),
        "v zádveří",
        ("light.zadveri", "light.ovladani_osvetleni_zadveri"),
    ),
    (
        ("v ložnici", "v loznici", "ložnice", "loznice"),
        "v ložnici",
        ("light.loznice", "light.ovladani_osvetleni_loznice"),
    ),
    (
        ("v koupelně", "v koupelne", "koupelna", "koupelně"),
        "v koupelně",
        ("light.koupelna", "light.ovladani_osvetleni_koupelna"),
    ),
    (
        ("na chodbě", "na chodbe", "v chodbě", "chodba", "chodbě"),
        "na chodbě",
        ("light.chodba", "light.ovladani_osvetleni_chodba"),
    ),
    (
        ("v pracovně", "v pracovne", "pracovna", "pracovně"),
        "v pracovně",
        ("light.pracovna", "light.ovladani_osvetleni_pracovna"),
    ),
    (
        ("v garáži", "v garazi", "garáž", "garaz", "garáži"),
        "v garáži",
        ("light.garaz", "light.ovladani_osvetleni_garaz"),
    ),
    (
        ("v šatně", "v satne", "šatna", "satna", "šatně"),
        "v šatně",
        ("light.satna", "light.ovladani_osvetleni_satna"),
    ),
)


def _expand_bare_area_command(hass: HomeAssistant, user_text: str) -> str:
    """Expand a bare Czech room utterance (e.g. 'v obýváku') into a state-aware light toggle command."""
    normalized = (user_text or "").strip().rstrip(".!?,").strip().lower()
    if not normalized:
        return user_text
    for aliases, canonical_phrase, entity_ids in _BARE_AREA_LIGHT_MAP:
        if normalized in aliases:
            any_on = False
            for eid in entity_ids:
                st = hass.states.get(eid)
                if st and st.state == "on":
                    any_on = True
                    break
            verb = "Zhasni" if any_on else "Rozsviť"
            expanded = f"{verb} světla {canonical_phrase}"
            LOGGER.info(
                "Expanded bare-area command %r -> %r (lights_on=%s)",
                user_text,
                expanded,
                any_on,
            )
            return expanded
    return user_text


def _build_smarthome_live_telemetry(hass: HomeAssistant) -> str:
    """Build an exact real-time Czech telemetry summary from sensor.deye_15kw_telemetry and hass.states."""
    def _st(eid: str, default: str = "neznámé") -> str:
        s = hass.states.get(eid)
        if not s or s.state in ("unknown", "unavailable"):
            return default
        unit = s.attributes.get("unit_of_measurement", "")
        return f"{s.state} {unit}".strip()

    deye_state = hass.states.get("sensor.deye_15kw_telemetry")
    deye_attrs: dict[str, Any] = dict(deye_state.attributes) if deye_state else {}

    def _da(key: str, unit: str = "", fallback_eid: str | None = None) -> str:
        val = deye_attrs.get(key)
        if val is not None and val != "":
            return f"{val} {unit}".strip() if unit else str(val)
        if fallback_eid:
            return _st(fallback_eid)
        return "neznámé"

    lines = [
        "\n--- AKTUÁLNÍ ŽIVÁ TELEMETRIE CHYTRÉ DOMÁCNOSTI ---",
        f"- Aktuální okamžitý příkon (spotřeba) domu právě teď: {_da('load_total_w', 'W', 'sensor.spotreba_domu_celkem')} (Fáze L1: {_da('load_l1_w', 'W', 'sensor.spotreba_domu_faze_l1')}, Fáze L2: {_da('load_l2_w', 'W', 'sensor.spotreba_domu_faze_l2')}, Fáze L3: {_da('load_l3_w', 'W', 'sensor.spotreba_domu_faze_l3')})",
        f"- Denní spotřeba domu dnes (od půlnoci): {_da('day_load_kwh', 'kWh', 'sensor.spotreba_domu_dnes')} (Celková kumulativní spotřeba domu: {_da('total_load_kwh', 'kWh', 'sensor.spotreba_domu_celkem_kwh')})",
        f"- Okamžitý odběr / přetok z distribuční sítě (Eastron) právě teď: {_da('grid_power_total_w', 'W', 'sensor.sit_aktualni_odber_a_pretok')} ({_da('grid_flow_state', '', 'sensor.sit_stav_toku_energie')})",
        f"- Dnešní odběr ze sítě (denní nákup ze sítě od půlnoci): {_da('day_grid_buy_kwh', 'kWh', 'sensor.sit_odber_dnes')} (Celkem nakoupeno ze sítě: {_da('total_grid_buy_kwh', 'kWh')}, Dnes prodáno do sítě: {_da('day_grid_sell_kwh', 'kWh')})",
        f"- Výkon solární elektrárny FVE Deye 15kW právě teď: {_da('pv_total_power_w', 'W', 'sensor.fve_celkovy_vykon')} (Dnes vyrobeno FVE: {_da('day_pv_kwh', 'kWh', 'sensor.fve_vyroba_dnes')}, Celkem vyrobeno FVE: {_da('total_pv_kwh', 'kWh', 'sensor.fve_vyroba_celkem')})",
        f"- Bateriové úložiště FVE Deye: nabití {_da('battery_soc_pct', '%', 'sensor.baterie_stav_nabiti')}, výkon {_da('battery_power_w', 'W', 'sensor.baterie_aktualni_vykon')}, stav: {_da('battery_status', '', 'sensor.baterie_stav')} (fyzická baterie k FVE zatím není pořízena, proto je 0 %)",
        f"- Venkovní počasí: teplota {_st('sensor.smarthome_venkovni_teplota')}, pocitová {_st('sensor.smarthome_pocitova_teplota')}, vlhkost {_st('sensor.smarthome_venkovni_vlhkost')}, vítr {_st('sensor.smarthome_rychlost_vetru')}, predikce FVE dnes {_st('sensor.smarthome_predikce_fve_dnes')} a zítra {_st('sensor.smarthome_predikce_fve_zitra')}",
        f"- Robotický vysavač: {_st('vacuum.smart_robot_vacuum')}",
        f"- Stav světel: obývák a kuchyň {_st('light.ovladani_osvetleni_kuchyne')}, zádveří {_st('light.zadveri')}, ložnice {_st('light.loznice')}, koupelna {_st('light.koupelna')}, chodba {_st('light.chodba')}, pracovna {_st('light.pracovna')}, garáž {_st('light.garaz')}, šatna {_st('light.satna')}",
        "PRAVIDLA PRO ODPOVĚĎ:",
        "1. Všechna čísla v odpovědi piš vždy číslicemi s jednotkou (např. 530 W, 14,0 kWh, 8,4 °C) — nikdy nepiš stovky nebo desítky slovy, aby nedošlo k záměně 530 W za 53 W.",
        "2. Rozlišuj přesně mezi 'Denní spotřeba domu dnes (od půlnoci)' v kWh (např. 14,0 kWh) a 'Aktuální okamžitý příkon (spotřeba) domu právě teď' ve W (např. 530 W). Při dotazu na denní spotřebu domu vždy uveď nejprve dnešní spotřebu v kWh a doplň i aktuální okamžitý příkon ve W.",
        "3. Na běžné konverzační dotazy a pozdravy (např. 'Ahoj, jak se máš?', 'Dobrý den') vždy přátelsky a stručně odpověz česky.",
        "4. Pokud uživatel řekne jen název místnosti (např. 'v obýváku') nebo požádá o rozsvícení/zhasnutí v obýváku, okamžitě přepni světlo v dané místnosti (v obýváku a kuchyni je to entita light.ovladani_osvetleni_kuchyne).",
    ]
    return "\n".join(lines)


async def _async_bidi_live_process(
    hass: HomeAssistant,
    api_key: str,
    model_name: str,
    prompt: str,
    user_text: str,
    llm_api: llm.APIInstance | None,
    temperature: float,
) -> str:
    """Execute a turn against Gemini 3.8 Live via WebSocket BidiGenerateContent."""
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    url = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        f"?key={api_key}"
    )
    session = async_get_clientsession(hass)

    bidi_tools: list[dict[str, Any]] = []
    if llm_api and llm_api.tools:
        f_decls = [
            _format_bidi_tool(tool, llm_api.custom_serializer)
            for tool in llm_api.tools
        ]
        if f_decls:
            bidi_tools = [{"functionDeclarations": f_decls}]

    async def _recv_json(ws: aiohttp.ClientWebSocketResponse) -> dict[str, Any]:
        msg = await ws.receive(timeout=25)
        if msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
            raw = msg.data.decode("utf-8") if isinstance(msg.data, bytes) else msg.data
            return json.loads(raw)
        raise HomeAssistantError(f"Unexpected WebSocket frame from Gemini Live: {msg}")

    async with session.ws_connect(url, timeout=25) as ws:
        setup_payload: dict[str, Any] = {
            "setup": {
                "model": model_name,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "temperature": temperature,
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": "Aoede"
                            }
                        }
                    },
                },
                "outputAudioTranscription": {},
                "systemInstruction": {
                    "parts": [{"text": prompt + _build_smarthome_live_telemetry(hass)}]
                },
            }
        }
        if bidi_tools:
            setup_payload["setup"]["tools"] = bidi_tools

        await ws.send_json(setup_payload)
        await _recv_json(ws)

        await ws.send_json(
            {
                "clientContent": {
                    "turns": [{"role": "user", "parts": [{"text": user_text}]}],
                    "turnComplete": True,
                }
            }
        )

        transcript_parts: list[str] = []
        text_parts: list[str] = []
        pcm_bytes = bytearray()

        for _ in range(120):
            data = await _recv_json(ws)
            if "toolCall" in data and llm_api:
                fcalls = data["toolCall"].get("functionCalls", [])
                fresps = []
                for fc in fcalls:
                    t_name = fc.get("name", "")
                    t_args = _escape_decode(fc.get("args", {}))
                    t_input = llm.ToolInput(tool_name=t_name, tool_args=t_args)
                    try:
                        raw_res = await llm_api.async_call_tool(t_input)
                        if isinstance(raw_res, dict):
                            f_res = raw_res
                        elif hasattr(raw_res, "data") or hasattr(raw_res, "error"):
                            f_res = {
                                "data": getattr(raw_res, "data", None),
                                "error": getattr(raw_res, "error", None),
                            }
                        else:
                            f_res = {"result": str(raw_res)}
                    except (HomeAssistantError, vol.Invalid, Exception) as err:
                        f_res = {"error": type(err).__name__, "error_text": str(err)}
                    fresps.append(
                        {
                            "id": fc.get("id"),
                            "name": t_name,
                            "response": json.loads(json.dumps({"output": f_res}, default=str)),
                        }
                    )
                await ws.send_json({"toolResponse": {"functionResponses": fresps}})
                continue

            sc = data.get("serverContent", {})
            if "outputTranscription" in sc:
                t = sc["outputTranscription"].get("text", "")
                if t:
                    transcript_parts.append(t)
            mt = sc.get("modelTurn", {})
            for p in mt.get("parts", []):
                if "text" in p:
                    text_parts.append(p["text"])
                inline = p.get("inlineData", {})
                if inline.get("mimeType", "").startswith("audio/pcm") and "data" in inline:
                    pcm_bytes.extend(base64.b64decode(inline["data"]))
            if sc.get("turnComplete"):
                break

        if pcm_bytes:
            def _write_wav(raw_pcm: bytes) -> None:
                www_dir = Path(hass.config.path("www"))
                www_dir.mkdir(parents=True, exist_ok=True)
                wav_path = www_dir / "gemini_live_reply.wav"
                with wave.open(str(wav_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(raw_pcm)

            await hass.async_add_executor_job(_write_wav, bytes(pcm_bytes))

        final_text = "".join(transcript_parts).strip()
        if not final_text:
            final_text = " ".join(text_parts).strip()
        return final_text or "Hotovo."


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        async_add_entities(
            [GoogleGenerativeAIConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class GoogleGenerativeAIConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    GoogleGenerativeAILLMBaseEntity,
):
    """Google Generative AI conversation agent."""

    _attr_supports_streaming = False

    def __init__(self, entry: ConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    @override
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    @override
    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    @override
    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @override
    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""
        options = self.subentry.data
        cleaned_input = (user_input.text or "").strip().lower()
        if cleaned_input in ("", "<silence>", "<noise>", "[silence]", "[noise]"):
            silent_resp = intent.IntentResponse(language=user_input.language)
            silent_resp.async_set_speech("Neslyšel jsem žádný příkaz.")
            return conversation.ConversationResult(
                response=silent_resp,
                conversation_id=chat_log.conversation_id,
                continue_conversation=False,
            )

        effective_user_text = _expand_bare_area_command(self.hass, user_input.text)

        try:
            base_prompt = options.get(CONF_PROMPT) or ""
            full_prompt = base_prompt + _build_smarthome_live_telemetry(self.hass)
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                full_prompt,
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        model_name = str(options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL))
        if "live" in model_name.lower():
            prompt_content = chat_log.content[0] if chat_log.content else None
            prompt = getattr(prompt_content, "content", "") or ""
            try:
                speech_text = await _async_bidi_live_process(
                    self.hass,
                    self.entry.data[CONF_API_KEY],
                    model_name,
                    prompt,
                    effective_user_text,
                    chat_log.llm_api,
                    float(options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)),
                )
            except Exception as err:
                LOGGER.error("Error in Gemini 3.8 Live BidiGenerateContent: %s", err)
                err_resp = intent.IntentResponse(language=user_input.language)
                err_resp.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Chyba při komunikaci s Gemini 3.8 Live: {err}",
                )
                return conversation.ConversationResult(
                    response=err_resp,
                    conversation_id=chat_log.conversation_id,
                )
            ok_resp = intent.IntentResponse(language=user_input.language)
            ok_resp.async_set_speech(speech_text)
            return conversation.ConversationResult(
                response=ok_resp,
                conversation_id=chat_log.conversation_id,
            )

        try:
            await self._async_handle_chat_log(chat_log)
            res = conversation.async_get_result_from_chat_log(user_input, chat_log)
            res.continue_conversation = False
            return res
        except Exception as primary_err:
            LOGGER.warning(
                "Primary conversation model %s failed (%s), falling back to Gemini 3.8 Live Bidi...",
                model_name,
                primary_err,
            )
            prompt_content = chat_log.content[0] if chat_log.content else None
            prompt = getattr(prompt_content, "content", "") or ""
            speech_text = await _async_bidi_live_process(
                self.hass,
                self.entry.data[CONF_API_KEY],
                "models/gemini-3.8-live",
                prompt,
                effective_user_text,
                chat_log.llm_api,
                float(options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)),
            )
            ok_resp = intent.IntentResponse(language=user_input.language)
            ok_resp.async_set_speech(speech_text)
            return conversation.ConversationResult(
                response=ok_resp,
                conversation_id=chat_log.conversation_id,
                continue_conversation=False,
            )
