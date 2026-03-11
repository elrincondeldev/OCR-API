import { algoliasearch } from "algoliasearch";
import crypto from "crypto";
import { AlgoliaMatch } from "../models.js";

let _client: ReturnType<typeof algoliasearch> | null = null;

function getClient() {
  if (!_client) {
    _client = algoliasearch(
      process.env.ALGOLIA_APP_ID!,
      process.env.ALGOLIA_API_KEY!
    );
    console.info(`Algolia client initialised (app_id=${process.env.ALGOLIA_APP_ID}).`);
  }
  return _client;
}

export async function searchInventory(
  queryText: string
): Promise<AlgoliaMatch | null> {
  try {
    const response = await getClient().searchSingleIndex({
      indexName: process.env.ALGOLIA_INDEX_NAME!,
      searchParams: {
        query: queryText,
        hitsPerPage: 1,
        attributesToRetrieve: ["item_name", "standard_quantity"],
        attributesToHighlight: [],
      },
    });

    if (!response.hits.length) {
      console.warn(`Algolia returned no hits for query: '${queryText}'`);
      return null;
    }

    const hit = response.hits[0] as Record<string, unknown> & {
      objectID: string;
    };
    const { objectID, item_name, standard_quantity, _highlightResult, ...rest } =
      hit;

    console.info(
      `Algolia matched '${queryText}' → '${item_name}' (objectID=${objectID})`
    );
    return {
      object_id: objectID,
      matched_item_name: (item_name as string) ?? "",
      standard_quantity: standard_quantity as string | undefined,
      metadata: rest as Record<string, unknown>,
    };
  } catch (err) {
    console.error(`Algolia search error for '${queryText}':`, err);
    return null;
  }
}

export async function searchFlaggedLines(
  flaggedTexts: string[]
): Promise<(AlgoliaMatch | null)[]> {
  return Promise.all(flaggedTexts.map(searchInventory));
}

export async function saveIngredients(
  ingredientNames: string[]
): Promise<number> {
  if (!ingredientNames.length) return 0;

  const seen = new Set<string>();
  const objects: object[] = [];

  for (const name of ingredientNames) {
    const normalised = name.toLowerCase().trim();
    if (!normalised || seen.has(normalised)) continue;
    seen.add(normalised);
    const objectID = crypto
      .createHash("md5")
      .update(normalised)
      .digest("hex");
    objects.push({ objectID, item_name: name.trim() });
  }

  if (!objects.length) return 0;

  await getClient().saveObjects({
    indexName: process.env.ALGOLIA_INDEX_NAME!,
    objects,
  });

  console.info(
    `Upserted ${objects.length} ingredient(s) to Algolia index '${process.env.ALGOLIA_INDEX_NAME}'.`
  );
  return objects.length;
}
