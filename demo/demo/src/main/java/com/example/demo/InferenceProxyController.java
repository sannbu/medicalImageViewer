package com.example.demo;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.net.ConnectException;

@RestController
public class InferenceProxyController {

    private static final String CRLF = "\r\n";

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String yoloInferUrl;
    private final Duration requestTimeout;

    public InferenceProxyController(
            ObjectMapper objectMapper,
            @Value("${proxy.yolo.infer-url:http://127.0.0.1:5000/infer}") String yoloInferUrl,
            @Value("${proxy.yolo.timeout-seconds:120}") long timeoutSeconds
    ) {
        this.objectMapper = objectMapper;
        this.yoloInferUrl = yoloInferUrl;
        this.requestTimeout = Duration.ofSeconds(Math.max(1L, timeoutSeconds));
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    @PostMapping(
            path = "/infer",
            consumes = MediaType.MULTIPART_FORM_DATA_VALUE,
            produces = MediaType.APPLICATION_JSON_VALUE
    )
    public ResponseEntity<?> infer(@RequestParam("image") MultipartFile image) {
        if (image == null || image.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "form-data field 'image' is required"));
        }

        try {
            String boundary = "----SpringProxyBoundary" + UUID.randomUUID().toString().replace("-", "");
            byte[] multipartBody = buildMultipartBody(image, boundary);

            HttpRequest upstreamRequest = HttpRequest.newBuilder(URI.create(yoloInferUrl))
                    .header(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE)
                    .header(HttpHeaders.CONTENT_TYPE, "multipart/form-data; boundary=" + boundary)
                    .timeout(requestTimeout)
                    .POST(HttpRequest.BodyPublishers.ofByteArray(multipartBody))
                    .build();

            HttpResponse<String> upstreamResponse = httpClient.send(
                    upstreamRequest,
                    HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8)
            );

            int upstreamStatus = upstreamResponse.statusCode();
            String rawBody = upstreamResponse.body() == null ? "" : upstreamResponse.body().trim();
            if (rawBody.isEmpty()) {
                return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body(Map.of(
                        "error", "YOLO service returned an empty response",
                        "upstream_status", upstreamStatus
                ));
            }

            JsonNode jsonBody;
            try {
                jsonBody = objectMapper.readTree(rawBody);
            } catch (Exception jsonEx) {
                return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body(Map.of(
                        "error", "YOLO service returned non-JSON response",
                        "upstream_status", upstreamStatus,
                        "body_preview", abbreviate(rawBody, 280)
                ));
            }

            HttpStatusCode statusCode = HttpStatusCode.valueOf(upstreamStatus);
            return ResponseEntity.status(statusCode)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(jsonBody);
        } catch (ConnectException connectEx) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body(Map.of(
                    "error", "Could not connect to YOLO service",
                    "detail", safeMessage(connectEx),
                    "upstream_url", yoloInferUrl
            ));
        } catch (IOException ioEx) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body(Map.of(
                    "error", "Proxy I/O failure while forwarding to YOLO service",
                    "detail", safeMessage(ioEx),
                    "upstream_url", yoloInferUrl
            ));
        } catch (InterruptedException interruptedEx) {
            Thread.currentThread().interrupt();
            return ResponseEntity.status(HttpStatus.GATEWAY_TIMEOUT).body(Map.of(
                    "error", "Proxy request interrupted",
                    "detail", safeMessage(interruptedEx)
            ));
        } catch (Exception ex) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body(Map.of(
                    "error", "Proxy to YOLO service failed",
                    "detail", safeMessage(ex)
            ));
        }
    }

    private byte[] buildMultipartBody(MultipartFile image, String boundary) throws IOException {
        String filename = Optional.ofNullable(image.getOriginalFilename())
                .filter(name -> !name.isBlank())
                .orElse("upload.png");
        String contentType = Optional.ofNullable(image.getContentType())
                .filter(type -> !type.isBlank())
                .orElse(MediaType.APPLICATION_OCTET_STREAM_VALUE);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(("--" + boundary + CRLF).getBytes(StandardCharsets.UTF_8));
        out.write((
                "Content-Disposition: form-data; name=\"image\"; filename=\"" + escapeQuotes(filename) + "\"" + CRLF
        ).getBytes(StandardCharsets.UTF_8));
        out.write(("Content-Type: " + contentType + CRLF + CRLF).getBytes(StandardCharsets.UTF_8));
        out.write(image.getBytes());
        out.write(CRLF.getBytes(StandardCharsets.UTF_8));
        out.write(("--" + boundary + "--" + CRLF).getBytes(StandardCharsets.UTF_8));
        return out.toByteArray();
    }

    private String escapeQuotes(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private String abbreviate(String text, int limit) {
        if (text == null || text.length() <= limit) {
            return text;
        }
        return text.substring(0, limit) + "...";
    }

    private String safeMessage(Throwable throwable) {
        if (throwable == null) {
            return "unknown error";
        }
        String message = throwable.getMessage();
        if (message == null || message.isBlank()) {
            return throwable.getClass().getSimpleName();
        }
        return message;
    }
}
